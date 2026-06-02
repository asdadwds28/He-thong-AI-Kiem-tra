import numpy as np
import cv2
from itertools import combinations

class GeometryUtils:
    @staticmethod
    def identify_slots_logic(centers):
        """
        Thuật toán SVD + Cross Product + Projection 
        Xác định Slot 1-5 bất kể khay xoay 360 độ.
        """
        centers = np.array(centers, dtype=np.float32)
        if len(centers) != 5:
            return None

        indices = list(range(5))
        
        # 1. Tìm hàng 3 vật thể (Group 3) dựa trên độ thẳng hàng (SVD)
        best_err = float('inf')
        group3_idx = []
        for idx3 in combinations(indices, 3):
            p3 = centers[list(idx3)]
            mean = np.mean(p3, axis=0)
            _, s, _ = np.linalg.svd(p3 - mean)
            if s[1] < best_err:
                best_err = s[1]
                group3_idx = list(idx3)
        
        group2_idx = [x for x in indices if x not in group3_idx]
        
        # 2. Xác định Vector Ngang (V_side): Từ tâm hàng 3 hướng sang tâm hàng 2
        center3 = np.mean(centers[group3_idx], axis=0)
        center2 = np.mean(centers[group2_idx], axis=0)
        v_side = center2 - center3 

        # 3. Xác định Vector Dọc tạm thời của hàng 3
        p3_pts = centers[group3_idx]
        dist_m = np.linalg.norm(p3_pts[:, None, :] - p3_pts[None, :, :], axis=-1)
        i_end1, i_end2 = np.unravel_index(np.argmax(dist_m), dist_m.shape)
        
        p_a = p3_pts[i_end1]
        p_b = p3_pts[i_end2]
        v_temp = p_b - p_a 
        
        # 4. Sử dụng Tích có hướng (Cross Product) để định hướng 1-2-3
        # Quy ước hệ tọa độ OpenCV (Y xuống): Nếu 1->3, hàng 2 nằm bên PHẢI thì cp > 0
        cp = v_temp[0] * v_side[1] - v_temp[1] * v_side[0]
        
        if cp > 0:
            s1_idx, s3_idx = group3_idx[i_end1], group3_idx[i_end2]
        else:
            s1_idx, s3_idx = group3_idx[i_end2], group3_idx[i_end1]

        # Slot 2 nằm giữa 1 và 3
        s2_idx = [x for x in group3_idx if x not in [s1_idx, s3_idx]][0]
        
        # 5. Xác định Slot 4, 5 bằng cách chiếu lên trục 1 -> 3
        v_spine = centers[s3_idx] - centers[s1_idx]
        v_spine /= (np.linalg.norm(v_spine) + 1e-6) # Tránh chia cho 0
        
        proj_g2 = []
        for i2 in group2_idx:
            proj_val = np.dot(centers[i2] - centers[s1_idx], v_spine)
            proj_g2.append((i2, proj_val))
        
        # Sắp xếp theo hình chiếu: gần Slot 1 hơn là Slot 4, gần Slot 3 hơn là Slot 5
        proj_g2.sort(key=lambda x: x[1])
        s4_idx, s5_idx = proj_g2[0][0], proj_g2[1][0]

        # Trả về Dictionary theo đúng format cũ để processor.py không bị lỗi
        return {
            1: centers[s1_idx].astype(int),
            2: centers[s2_idx].astype(int),
            3: centers[s3_idx].astype(int),
            4: centers[s4_idx].astype(int),
            5: centers[s5_idx].astype(int)
        }

    @staticmethod
    def calculate_iou_polygon(box_item, poly_slot):
        """Giữ nguyên hàm này vì nó đang hoạt động tốt cho va chạm"""
        x1, y1, x2, y2 = box_item
        poly_item = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        poly_slot = poly_slot.astype(np.float32)

        item_area = (x2 - x1) * (y2 - y1)
        if item_area <= 0: return 0.0, 0.0, 0.0

        ret, intersect_pts = cv2.intersectConvexConvex(poly_item, poly_slot)
        intersection_area = cv2.contourArea(intersect_pts) if (ret and intersect_pts is not None) else 0.0
        
        slot_area = cv2.contourArea(poly_slot)
        denominator = min(item_area, slot_area)
        
        if denominator <= 0: return 0.0, 0.0, 0.0

        ratio = intersection_area / denominator
        return intersection_area, item_area, ratio

    @staticmethod
    def is_item_in_slot(box_item, poly_slot, threshold=0.45): 
        _, _, ratio = GeometryUtils.calculate_iou_polygon(box_item, poly_slot)
        return ratio >= threshold