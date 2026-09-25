import cv2
import numpy as np

class TrafficLightDetector:
    def __init__(self):
        self.state_history = []
        self.last_stable_state = "UNKNOWN"
        self.frame_counter = 0

    def get_state(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
        self.frame_counter += 1
        
        # Calculate state only every 3rd frame
        if self.frame_counter % 3 != 1:
            if not self.state_history:
                return "GREEN" # default fallback
            return self.last_stable_state if self.last_stable_state != "UNKNOWN" else "GREEN"

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            raw_state = "UNKNOWN"
        else:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raw_state = "UNKNOWN"
            else:
                raw_state = self._analyze_crop(crop)

        self.state_history.append(raw_state)
        # Keep last 12 states (~0.4s at 30fps)
        if len(self.state_history) > 12:
            self.state_history.pop(0)

        # Hysteresis: state changes only if stable for >= 12 frames
        if len(self.state_history) == 12 and all(s == raw_state for s in self.state_history):
            if raw_state != "UNKNOWN":
                self.last_stable_state = raw_state

        return self.last_stable_state if self.last_stable_state != "UNKNOWN" else "GREEN"

    def _analyze_crop(self, crop: np.ndarray) -> str:
        h, w = crop.shape[:2]
        
        # Divide into 3 ROIs: top, middle, bottom
        h3 = h // 3
        rois = {
            "RED": crop[0:h3, :],
            "YELLOW": crop[h3:2*h3, :],
            "GREEN": crop[2*h3:h, :]
        }

        scores = {}
        for color, roi in rois.items():
            if roi.size == 0:
                scores[color] = 0
                continue

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)

            # Brightness threshold
            v_mask = V > 170

            if color == "RED":
                # Hue [0,12] or [165,180] and S > 90
                h_mask = ((H >= 0) & (H <= 12)) | ((H >= 165) & (H <= 180))
                s_mask = S > 90
            elif color == "GREEN":
                # Hue [60,100] and S > 60
                h_mask = (H >= 60) & (H <= 100)
                s_mask = S > 60
            elif color == "YELLOW":
                # Hue [12,35]
                h_mask = (H >= 12) & (H <= 35)
                # For yellow we can just use V > 170 and S > 90 to be safe
                s_mask = S > 90

            final_mask = v_mask & h_mask & s_mask
            score = np.count_nonzero(final_mask) / (roi.shape[0] * roi.shape[1] + 1e-6)
            scores[color] = score

        best_color = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_color]

        min_on = 0.05 # At least 5% of the ROI should be lit up
        if best_score > min_on:
            if best_color == "YELLOW":
                # In Part A rules, Yellow is generally treated as GREEN (or not RED). 
                # The prompt says: "Sariq (YELLOW) qizilga tenglashtirilmaydi."
                return "YELLOW"
            return best_color
        
        return "UNKNOWN"
