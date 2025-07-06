import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import random
import ctypes
import pycuda.driver as cuda
import time

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# Buffers untuk host/device
host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
bindings = []


class YoloTRTSeg():
    def __init__(self, library, engine_path, conf, yolo_ver):
        self.CONF_THRESH = conf
        self.IOU_THRESHOLD = 0.1
        self.yolo_version = yolo_ver
        self.categories = ["Dummy", "K34", "Korban", "Turu"]
        self.num_classes = len(self.categories)
        self.class_colors = {
            "Dummy": [0, 125, 0],
            "K34": [200, 0, 0],
            "Korban": [0, 0, 200],
            "Turu": [150, 150, 0],
        }
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        ctypes.CDLL(library)
        with open(engine_path, 'rb') as f:
            serialized = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self.batch_size = 1

        # Inisialisasi placeholder
        self.idx_input = -1
        self.idx_output = -1
        self.idx_proto = -1
        self.output_binding_order = {}  # Tambahkan ini

        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(binding)
            print(f"[DEBUG] Binding name: '{binding}', idx={idx}, shape={shape}")
            size = trt.volume(shape) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))

            if self.engine.binding_is_input(binding):
                # Binding input (data)
                self.idx_input = idx
                if len(shape) == 4:
                    _, c, h, w = shape
                else:
                    c, h, w = shape
                self.input_h = int(h)
                self.input_w = int(w)
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                print(f"[INFO] Network expects input = {self.input_w}×{self.input_h}")
            else:
                # Binding output (bisa prob → deteksi, atau proto → masker)
                pos = len(host_outputs)
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                self.output_binding_order[idx] = pos

                name = binding.lower()
                if "prob" in name:
                    # Karena binding untuk deteksi keluaran bernama "prob"
                    self.idx_output = idx
                elif "proto" in name:
                    self.idx_proto = idx

        if self.idx_output < 0 or self.idx_proto < 0:
            raise RuntimeError("Engine tidak ditemukan binding 'output' atau 'proto'.")

        # Ambil shape output dan shape proto
        shape_output = self.engine.get_binding_shape(self.idx_output)
        shape_proto = self.engine.get_binding_shape(self.idx_proto)

        # Misal N_MASK_COEFF = 32, LEN_ONE_RESULT = 6 + 32
        self.N_MASK_COEFF = 32
        self.LEN_ONE_RESULT = 6 + self.N_MASK_COEFF
        self.LEN_ALL_RESULT = 1 + 1000 * self.LEN_ONE_RESULT

        # Tentukan proto_dim, proto_h, proto_w dari shape_proto:
        if len(shape_proto) == 4:
            _, self.proto_dim, self.proto_h, self.proto_w = shape_proto
        else:
            self.proto_dim, self.proto_h, self.proto_w = shape_proto

    def PreProcessImg(self, img):
        """
        Resize & pad image --> bentuk [1,3,input_h,input_w], RGB, float32 di [0,1]
        """
        image_raw = img
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1, tx2 = 0, 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1, ty2 = 0, 0

        image = cv2.resize(image, (tw, th))
        # Padding dengan warna abu‐abu (128,128,128)
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32) / 255.0
        # [H,W,C] --> [C,H,W]
        image = np.transpose(image, [2, 0, 1])
        # Tambah batch dim: [1,C,H,W]
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w, tx1, ty1, tw, th

    def Inference(self, img):
        # 1) Preprocess → isi host_inputs[0]
        input_image, image_raw, origin_h, origin_w, tx1, ty1, tw, th = self.PreProcessImg(img)
        np.copyto(host_inputs[0], input_image.ravel())

        # 2) Copy input → device
        stream = cuda.Stream()
        context = self.engine.create_execution_context()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)

        # 3) Eksekusi
        t1 = time.time()
        context.execute_async(self.batch_size, bindings, stream_handle=stream.handle)

        # 4) Copy hasil device→host
        pos_out = self.output_binding_order[self.idx_output]
        pos_proto = self.output_binding_order[self.idx_proto]

        cuda.memcpy_dtoh_async(host_outputs[pos_out], cuda_outputs[pos_out], stream)
        cuda.memcpy_dtoh_async(host_outputs[pos_proto], cuda_outputs[pos_proto], stream)
        stream.synchronize()
        t2 = time.time()
        # 5) Ambil array numpy:
        output_flat = host_outputs[pos_out]  # prediksi deteksi + koef
        proto_flat = host_outputs[pos_proto]  # data prototype (size = proto_dim*proto_h*proto_w)

        # 6) PostProcess
        result = self.PostProcessSegmentation(
            output_flat, proto_flat, image_raw, origin_h, origin_w, tx1, ty1, tw, th
        )
        return result, 1/(t2 - t1)

    def PostProcessSegmentation(self, output_flat, proto_flat, image_raw, origin_h, origin_w, tx1, ty1, tw, th):
        """
        1) Baca jumlah deteksi (output_flat[0])
        2) Reshape prediksi ke [-1, LEN_ONE_RESULT] dan ambil baris pertama sampai num_det
        3) Ekstrak bbox, score, class, mask_coeffs
        4) Lakukan NMS (mirip detection) menggunakan 4 kolom pertama + confidence
        5) Untuk setiap deteksi sisa, buat masker:
           - Ambil koefisien: mask_coeffs (length N_MASK_COEFF)
           - Proto_flat reshape ke [mask_dim, proto_h, proto_w]
           - Hitung mask = sigmoid(sum_k(proto[k] * coeff[k]))
           - Resize mask ke bounding box asli pada gambar (origin_h,origin_w)
           - Threshold (misal >0.5)
        6) Ambil top‐3 deteksi (jika lebih dari 3), sesuai bounding box confidence
        7) Overlay kotak + masker berwarna ke image_raw
        8) Kembalikan list dict: tiap dict {"class", "conf", "box", "mask"}
        """
        # --------------- A) Ekstrak deteksi ---------------
        num_det = int(output_flat[0])
        # reshape prediksi: [-1, LEN_ONE_RESULT], ambil baris 1..num_det
        preds = np.reshape(output_flat[1:], (-1, self.LEN_ONE_RESULT))[:num_det, :]
        # preds[:, 0:4] = x,y,w,h (center), preds[:,4]=conf, preds[:,5]=class_id
        # preds[:,6:] = mask coefficients

        # Ubah xywh ke xyxy skala pre-pad untuk NMS (belum ke scale final)
        # Namun agar NMS tetap benar, serahkan langsung xywh (pada skala input) lalu ubah nanti
        # Kita akan implementasi NMS seperti detection: ubah ke xyxy *sebelum* scaling ke gambar asli
        # Gunakan metode yang mirip dengan NonMaxSuppression di deteksi
        # Tapi pertama, siapkan array untuk NMS: [x1,y1,x2,y2,conf,class_id, *mask_coeffs]
        boxes_input = []
        for row in preds:
            cx, cy, w, h, conf, cls_id = row[0:6]
            mask_coeffs = row[6:]  # length N_MASK_COEFF
            # Konversi xywh --> xyxy di ruang pre-pad
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_input.append([x1, y1, x2, y2, conf, cls_id] + mask_coeffs.tolist())
        boxes_input = np.array(boxes_input)

        # --------------- B) Lakukan clipping & scaling coordinates ke gambar asli ---------------
        # Fungsi bantu: konversi xyxy pre-pad ke koordinat gambar asli
        def scale_coords_prepad(x1, y1, x2, y2):
            """
            Masukkan x1,y1,x2,y2 pada skala (0..input_w,input_h) termasuk padding
            Keluarkan koordinat pada (0..origin_w,0..origin_h)
            Dengan memperhatikan offset tx1,ty1 dan ratio scaling (r_w, r_h)
            """
            # Tentukan ratio: sama dengan di PreProcessImg
            r_w = self.input_w / origin_w
            r_h = self.input_h / origin_h
            if r_h > r_w:
                # Dibandingkan, input width skala lebih sempit → ambil r_w sebagai scale
                scale = r_w
                pad_w = 0
                pad_h = (self.input_h - r_w * origin_h) / 2
            else:
                scale = r_h
                pad_w = (self.input_w - r_h * origin_w) / 2
                pad_h = 0
            # Hilangkan padding lalu bagi scale
            x1_n = (x1 - pad_w) / scale
            y1_n = (y1 - pad_h) / scale
            x2_n = (x2 - pad_w) / scale
            y2_n = (y2 - pad_h) / scale
            # Clip ke tepi
            x1_n = np.clip(x1_n, 0, origin_w - 1)
            y1_n = np.clip(y1_n, 0, origin_h - 1)
            x2_n = np.clip(x2_n, 0, origin_w - 1)
            y2_n = np.clip(y2_n, 0, origin_h - 1)
            return x1_n, y1_n, x2_n, y2_n

        # Lakukan NMS manual (sama dengan deteksi, tapi kita bawa kelas dan conf untuk sorting)
        # Urutkan descending berdasarkan conf
        if boxes_input.shape[0] == 0:
            return {
                "detections": [],
                "vis_image": image_raw.copy()
            }

        # Filter prediksi di bawah confidence threshold
        mask_conf_mask = boxes_input[:, 4] >= self.CONF_THRESH
        boxes_input = boxes_input[mask_conf_mask]

        # Jika tidak ada yang lolos:
        if boxes_input.shape[0] == 0:
            return {
                "detections": [],
                "vis_image": image_raw.copy()
            }

        # Urutkan berdasar conf
        idxs = np.argsort(-boxes_input[:, 4])
        boxes_sorted = boxes_input[idxs]

        keep_boxes = []
        while boxes_sorted.shape[0]:
            # Ambil box tertinggi
            current = boxes_sorted[0]
            keep_boxes.append(current)
            if boxes_sorted.shape[0] == 1:
                break
            # Hitung IoU dengan sisa
            xx1 = np.maximum(current[0], boxes_sorted[:, 0])
            yy1 = np.maximum(current[1], boxes_sorted[:, 1])
            xx2 = np.minimum(current[2], boxes_sorted[:, 2])
            yy2 = np.minimum(current[3], boxes_sorted[:, 3])

            inter_w = np.clip(xx2 - xx1 + 1, 0, None)
            inter_h = np.clip(yy2 - yy1 + 1, 0, None)
            inter_area = inter_w * inter_h

            area1 = (current[2] - current[0] + 1) * (current[3] - current[1] + 1)
            area2 = (boxes_sorted[:, 2] - boxes_sorted[:, 0] + 1) * \
                    (boxes_sorted[:, 3] - boxes_sorted[:, 1] + 1)

            iou = inter_area / (area1 + area2 - inter_area + 1e-16)

            # Filter: IoU <= threshold atau beda kelas
            keep_mask = (iou <= self.IOU_THRESHOLD) | (current[5] != boxes_sorted[:, 5])
            boxes_sorted = boxes_sorted[keep_mask]
        keep_boxes = np.stack(keep_boxes, 0)

        # Batasi maksimal 3 deteksi
        if keep_boxes.shape[0] > 3:
            keep_boxes = keep_boxes[:3]

        # --------------- C) Bangun masker dari prototype ---------------
        # ubah proto_flat ke [mask_dim, proto_h, proto_w]
        proto = np.reshape(proto_flat, (self.proto_dim, self.proto_h, self.proto_w))
        # Lakukan sigmoid di proto atau nanti di coeff?
        # Proses tiap deteksi:
        det_results = []

        for det in keep_boxes:
            # det = [x1,y1,x2,y2,conf,cls_id, coeff_0, coeff_1, ..., coeff_31]
            x1_p, y1_p, x2_p, y2_p = det[0:4]
            conf = det[4]
            cls_id = int(det[5])
            coeffs = det[6:]  # length = N_MASK_COEFF

            # Skala kotak ke koordinat asli
            x1_o, y1_o, x2_o, y2_o = scale_coords_prepad(x1_p, y1_p, x2_p, y2_p)
            x1_o, y1_o, x2_o, y2_o = int(x1_o), int(y1_o), int(x2_o), int(y2_o)
            box_w = x2_o - x1_o
            box_h = y2_o - y1_o

            # Hitung mask: prototype [mask_dim,proto_h,proto_w] x coeffs [mask_dim]
            # 1) Linear combination: sum_k(proto[k] * coeffs[k])
            mask_pred = np.zeros((self.proto_h, self.proto_w), dtype=np.float32)
            for k in range(self.proto_dim):
                mask_pred += proto[k] * coeffs[k]
            # 2) Sigmoid --> mask probabilitas
            mask_pred = 1.0 / (1.0 + np.exp(-mask_pred))
            # 3) Resize mask ke ukuran awak: (proto_h,proto_w) --> (input_h,input_w)
            mask_pred = cv2.resize(mask_pred, (self.input_w, self.input_h))
            # 4) Hilangkan padding: crop sesuai bounding box di ruang input
            #    Tapi lebih mudah langsung resize ke ukuran kotak di gambar asli:
            #    Pertama crop region bounding box di ruang input:
            # Hitung koordinat di ruang input sebelum padding:
            #   x1_p,y1_p,x2_p,y2_p sudah di ruang input
            mask_crop = mask_pred[int(y1_p):int(y2_p), int(x1_p):int(x2_p)]
            # Resize crop ke [box_h, box_w]
            if box_h > 0 and box_w > 0 and mask_crop.size > 0:
                mask_resized = cv2.resize(mask_crop, (box_w, box_h))
            else:
                # jika ada kasus degenerate
                mask_resized = np.zeros((box_h, box_w), dtype=np.float32)

            # 5) Threshold: >0.5 --> bina binary mask
            mask_bin = (mask_resized > 0.5).astype(np.uint8)

            # 6) Tempelkan ke gambar asli: buat layer mask fullsize di origin_h×origin_w
            mask_full = np.zeros((origin_h, origin_w), dtype=np.uint8)
            mask_full[y1_o:y2_o, x1_o:x2_o] = mask_bin

            # Simpan hasil deteksi
            det_dict = {
                "class": self.categories[cls_id],
                "conf": float(conf),
                "box": [x1_o, y1_o, x2_o, y2_o],
                "mask": mask_full  # binary mask
            }
            det_results.append(det_dict)

        # --------------- D) Overlay kotak & mask ke image_raw ---------------
        # Gambarkan hasil ke salinan image_raw
        # … setelah build daftar det_results …
        image_vis = image_raw.copy()

        for det in det_results:
            cls_name = det["class"]
            box = det["box"]
            mask_full = det["mask"]  # binary mask shape = (H, W)

            # Pilih warna overlay sesuai kelas
            color = self.class_colors.get(cls_name, [random.randint(0, 255) for _ in range(3)])

            # 1) Buat full-color frame untuk overlay
            colored_mask_full = np.zeros_like(image_vis, dtype=np.uint8)
            colored_mask_full[:, :] = color

            # 2) Blend full-frame antara image_vis dan colored_mask_full
            blended_full = cv2.addWeighted(
                image_vis.astype(np.uint8),
                0.5,
                colored_mask_full.astype(np.uint8),
                0.5,
                0
            )

            # 3) Buat mask_bool3d yang ukurannya (H, W, 3)
            mask_full_bool = mask_full.astype(bool)  # (H, W)
            mask_bool3d = np.repeat(mask_full_bool[..., None], 3, axis=2)  # (H, W, 3)

            # 4) Salin hanya piksel-piksel yang di-mask dari blended_full ke image_vis
            image_vis[mask_bool3d] = blended_full[mask_bool3d]

            # 5) Gambar bounding box dan label di atas overlay
            x1, y1, x2, y2 = box
            tl = max(1, int(round(0.002 * (origin_h + origin_w) / 2)))
            cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)

            label = f"{cls_name}:{det['conf']:.2f}"
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = (x1 + t_size[0], y1 - t_size[1] - 3)
            cv2.rectangle(image_vis, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(image_vis, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # Ganti image_raw dengan image_vis kalau ingin menampilkan
        # Atau kembalikan kedua: (det_results, image_vis)
        return {
            "detections": det_results,
            "vis_image": image_vis
        }
