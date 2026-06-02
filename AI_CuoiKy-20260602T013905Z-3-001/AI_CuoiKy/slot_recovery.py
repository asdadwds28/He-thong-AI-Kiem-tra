import cv2
import numpy as np

class SlotRecovery:
    def __init__(self):
        # Lưu tọa độ 5 slot chuẩn (Reference)
        # Dạng: {1: [x,y], ..., 5: [x,y]}
        self.ref_slots = None 

    def update_reference(self, current_slots):
        """
        Gọi hàm này khi nhận diện đủ 5 slot để cập nhật mẫu chuẩn.
        Giúp hệ thống thích nghi nếu camera bị rung nhẹ.
        """
        self.ref_slots = current_slots.copy()

    def recover(self, detected_slots):
        """
        Input: Dict các slot tìm thấy (VD: chỉ có 1, 2, 4)
        Output: Dict đủ 5 slot (đã khôi phục 3, 5)
        """
        # 1. Nếu chưa có mẫu chuẩn thì chịu, trả về cái đang có
        if self.ref_slots is None:
            return detected_slots

        # 2. Tìm các điểm chung (Anchor points) giữa hiện tại và mẫu
        # Ví dụ: Thấy được 1, 2, 4 -> Common IDs = {1, 2, 4}
        common_ids = set(detected_slots.keys()) & set(self.ref_slots.keys())

        # Cần ít nhất 2 điểm để tính toán góc xoay và vị trí
        if len(common_ids) < 2:
            return detected_slots # Không đủ dữ kiện, trả về gốc

        # 3. Tạo 2 mảng tọa độ để so khớp
        src_pts = [] # Tọa độ trên mẫu chuẩn
        dst_pts = [] # Tọa độ hiện tại tìm thấy
        
        for sid in common_ids:
            src_pts.append(self.ref_slots[sid])
            dst_pts.append(detected_slots[sid])

        src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)

        # 4. [QUAN TRỌNG] Tính ma trận biến đổi (Affine Transformation)
        # Hàm này tìm ma trận M sao cho: src_pts * M ~ dst_pts
        # estimateAffinePartial2D xử lý được: Dịch chuyển + Xoay + Co giãn (Scale)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        if M is None:
            return detected_slots

        # 5. Khôi phục các điểm còn thiếu
        recovered_slots = detected_slots.copy()
        missing_ids = set(self.ref_slots.keys()) - set(detected_slots.keys())

        if not missing_ids:
            return recovered_slots

        # Lấy tọa độ các điểm thiếu từ mẫu chuẩn
        missing_pts_ref = []
        for mid in missing_ids:
            missing_pts_ref.append(self.ref_slots[mid])
        
        missing_pts_ref = np.array(missing_pts_ref, dtype=np.float32).reshape(-1, 1, 2)

        # Áp dụng ma trận M để biến đổi điểm thiếu từ Mẫu -> Hiện tại
        # Công thức: transform(src, M)
        recovered_pts = cv2.transform(missing_pts_ref, M)

        # Gán ngược lại vào kết quả
        for i, mid in enumerate(missing_ids):
            # Lấy tọa độ [0][0] và [0][1]
            pos = recovered_pts[i][0]
            recovered_slots[mid] = pos.astype(int)

        return recovered_slots