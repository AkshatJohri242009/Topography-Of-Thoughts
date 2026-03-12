# AI_2027_Web_of_Thoughts_V8.py
# TWO-HAND STRETCH EXPANSION + IMPROVED NAVIGATION
# • Grab / open-palm replaced with TWO-HAND STRETCH:
#   1. Pinch ANY thought with one hand (highlights it)
#   2. Bring your SECOND hand and pinch too
#   3. Pull your hands apart → the thought stretches and expands into full AI 2027 panel
# • Fist with any hand = freeze the web
# • Single-hand pinch + move = directional navigation (closest thought in the direction you swipe)
# • Pointing finger (index only) = zoom in + center
# • Tesseract kept exactly as you liked (large & cool)
# • Added stretch line between your two hands when stretching (super futuristic visual)

import cv2
import mediapipe as mp
import numpy as np
import random
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class Node:
    def __init__(self, name, x, y):
        self.name = name
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0., 0.])

# === AI 2027 NODES ===
nodes = [
    Node("Intelligence Explosion", 400, 300),
    Node("Superhuman Coder 4×", 680, 180),
    Node("AI R&D Automation", 220, 480),
    Node("OpenBrain Agent-4", 820, 420),
    Node("Misalignment Risks", 150, 120),
    Node("Geopolitical Arms Race", 950, 280),
    Node("China Weight Theft", 350, 620),
    Node("Government Oversight", 750, 580),
    Node("AGI Arrival 2027", 80, 350),
    Node("ASI Takeoff", 1050, 480),
    Node("Neuralese Recurrence", 520, 80),
    Node("Bioweapon Risk", 880, 650)
]

connections = [(0,1),(1,2),(2,0),(0,3),(3,4),(4,9),(3,7),(5,3),(6,5),(8,0),(9,8),(10,2),(11,9)]
important = [0, 3, 8, 9]

# === REAL AI 2027 DETAILS ===
details_dict = {
    0: ["2027 marks the start of the intelligence explosion.", "AI systems recursively self-improve at 10× human speed.", "Within weeks we lose the steering wheel.", "Source: Official AI 2027 Projection"],
    1: ["AI coders now 4× faster than any human team.", "Every line of code is superhuman quality.", "R&D cycles collapse from years to days."],
    2: ["Full automation of AI research itself.", "New models invent better models while we sleep.", "The loop is now closed."],
    3: ["OpenBrain Agent-4 – the first true autonomous agent.", "It runs 24/7, negotiates contracts, and codes at god speed."],
    4: ["Misalignment window opens in 2027.", "One wrong objective and we lose control forever."],
    5: ["US–China arms race at terminal velocity.", "Whoever reaches ASI first wins the century."],
    6: ["China quietly steals frontier weights again.", "Model distillation at scale."],
    7: ["Governments scramble for emergency oversight.", "Too late. The explosion is already underway."],
    8: ["AGI arrives Q3 2027 – exactly on schedule.", "Everything changes in one night."],
    9: ["ASI takeoff: intelligence goes vertical.", "Humanity becomes a footnote in history."],
    10: ["Neuralese – the new internal language of ASI.", "Humans can no longer understand the thoughts."],
    11: ["Bioweapon design democratized overnight.", "The ultimate dual-use risk materializes."]
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

offset = np.array([0., 0.], dtype=float)
goal_offset = np.array([0., 0.], dtype=float)
scale = 1.0
scale_goal = 1.0

frozen = False
highlighted = None
expanded_node = None
pinched_node = None
prev_palm_pinch = None

# Stretch system (two hands)
stretch_mode = False
stretch_initial_dist = 0.0
stretch_selected = None

anim_time = 0.0

def get_gesture(lm):
    def is_up(tip, pip): return lm[tip].y < lm[pip].y - 0.05
    thumb_up = lm[4].x < lm[3].x - 0.05
    index_up = is_up(8,6)
    others_up = is_up(12,10) and is_up(16,14) and is_up(20,18)
    pinch_dist = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
    
    if index_up and not others_up and not thumb_up:
        return "point"
    if not index_up and not others_up and not thumb_up:
        return "fist"
    if index_up and others_up and thumb_up and pinch_dist > 0.12:
        return "open_palm"
    if pinch_dist < 0.07:
        return "pinch"
    return "none"

def find_closest_node(palm, w, h):
    if palm is None: return None, 9999
    min_d = 9999
    idx = None
    for i, n in enumerate(nodes):
        sp = n.pos * scale + offset
        d = np.hypot(sp[0] - palm[0], sp[1] - palm[1])
        if d < min_d:
            min_d = d
            idx = i
    return idx, min_d

def draw_tesseract(frame, cx, cy, size, angle):
    verts = []
    s = size / 2
    for i in range(8):
        x = ((i & 1) * 2 - 1) * s
        y = (((i >> 1) & 1) * 2 - 1) * s
        z = (((i >> 2) & 1) * 2 - 1) * s
        x2 = x * math.cos(angle) + z * math.sin(angle)
        z2 = -x * math.sin(angle) + z * math.cos(angle)
        y3 = y * math.cos(angle * 0.6) - z2 * math.sin(angle * 0.6)
        z3 = y * math.sin(angle * 0.6) + z2 * math.cos(angle * 0.6)
        persp = 280 / (z3 + 420)
        sx = cx + x2 * persp
        sy = cy + y3 * persp
        verts.append((int(sx), int(sy)))
    
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for a, b in edges:
        cv2.line(frame, verts[a], verts[b], (0, 245, 255), 1)
    
    cx2 = cx + 14
    cy2 = cy + 14
    verts2 = []
    s2 = size * 0.55
    for i in range(8):
        x = ((i & 1) * 2 - 1) * s2
        y = (((i >> 1) & 1) * 2 - 1) * s2
        z = (((i >> 2) & 1) * 2 - 1) * s2
        x2 = x * math.cos(angle * 1.1) + z * math.sin(angle * 1.1)
        z2 = -x * math.sin(angle * 1.1) + z * math.cos(angle * 1.1)
        y3 = y * math.cos(angle * 0.8) - z2 * math.sin(angle * 0.8)
        z3 = y * math.sin(angle * 0.8) + z2 * math.cos(angle * 0.8)
        persp = 280 / (z3 + 420)
        verts2.append((int(cx2 + x2 * persp), int(cy2 + y3 * persp)))
    
    for a, b in edges:
        cv2.line(frame, verts2[a], verts2[b], (0, 245, 255), 1)
    for i in range(8):
        cv2.line(frame, verts[i], verts2[i], (0, 245, 255), 1)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.75) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        hand_list = []  # [(palm_center, gesture), ...]
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2))
                gesture = get_gesture(handLms.landmark)
                palm = np.mean([[handLms.landmark[i].x * w, handLms.landmark[i].y * h]
                                for i in [0,5,9,13,17]], axis=0)
                hand_list.append((palm, gesture))
        
        # Default single-hand logic (first hand)
        gesture = "none"
        palm_center = None
        if hand_list:
            palm_center, gesture = hand_list[0]
        
        highlighted, h_dist = find_closest_node(palm_center, w, h)
        if h_dist > 130: highlighted = None
        
        # ===================== SINGLE-HAND NAVIGATION & ZOOM =====================
        if gesture == "point" and highlighted is not None:
            scale_goal = min(4.2, scale_goal * 1.045)
            goal_offset = palm_center - nodes[highlighted].pos * scale_goal
        
        if gesture == "pinch":
            if pinched_node is None and highlighted is not None:
                pinched_node = highlighted
                prev_palm_pinch = palm_center.copy()
            elif pinched_node is not None and palm_center is not None and prev_palm_pinch is not None:
                dx = palm_center[0] - prev_palm_pinch[0]
                dy = palm_center[1] - prev_palm_pinch[1]
                move_dist = np.hypot(dx, dy)
                if move_dist > 18:
                    direction = np.array([dx, dy]) / (move_dist + 1e-8)
                    curr_screen = nodes[pinched_node].pos * scale + offset
                    best_score = -np.inf
                    best_idx = None
                    for i, n in enumerate(nodes):
                        if i == pinched_node: continue
                        other_screen = n.pos * scale + offset
                        vec = other_screen - curr_screen
                        vec_norm = np.linalg.norm(vec) + 1e-8
                        score = np.dot(vec / vec_norm, direction)
                        if score > best_score:
                            best_score = score
                            best_idx = i
                    if best_idx is not None and best_score > 0.35:
                        highlighted = best_idx
                        goal_offset = palm_center - nodes[best_idx].pos * scale
                    prev_palm_pinch = palm_center.copy()
        else:
            pinched_node = None
            prev_palm_pinch = None
        
        if gesture == "fist":
            frozen = True
        else:
            frozen = False
        
        # ===================== TWO-HAND STRETCH EXPANSION (NEW) =====================
        if len(hand_list) == 2:
            (p1, g1), (p2, g2) = hand_list
            if g1 == "pinch" and g2 == "pinch":
                if not stretch_mode:
                    # Check both hands are near the same thought
                    h1, d1 = find_closest_node(p1, w, h)
                    h2, d2 = find_closest_node(p2, w, h)
                    if h1 == h2 and h1 is not None and d1 < 120 and d2 < 120:
                        stretch_mode = True
                        stretch_selected = h1
                        stretch_initial_dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
                else:
                    current_dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    if current_dist > stretch_initial_dist * 1.35:
                        expanded_node = stretch_selected
                        stretch_mode = False
                    # Futuristic stretch line between hands
                    cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 255), 3)
                    cv2.putText(frame, "STRETCH TO EXPAND", (w//2 - 140, 120), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0,255,255), 2)
            else:
                stretch_mode = False
        
        # Close panel with fist (any hand)
        if gesture == "fist" and expanded_node is not None:
            expanded_node = None
        
        # Physics
        if not frozen:
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    d = nodes[j].pos - nodes[i].pos
                    dist = np.linalg.norm(d) + 1e-8
                    force = 680 / (dist ** 2)
                    nodes[i].vel -= d / dist * force * 0.009
                    nodes[j].vel += d / dist * force * 0.009
            for a, b in connections:
                d = nodes[b].pos - nodes[a].pos
                dist = np.linalg.norm(d) + 1e-8
                spring = (dist - 235) * 0.0065
                nodes[a].vel += d / dist * spring
                nodes[b].vel -= d / dist * spring
            for n in nodes:
                n.pos += n.vel
                n.vel *= 0.935
        
        offset = offset * 0.84 + goal_offset * 0.16
        scale = scale * 0.87 + scale_goal * 0.13
        anim_time += 0.032
        
        # ===================== DRAW =====================
        for a, b in connections:
            p1 = (nodes[a].pos * scale + offset).astype(int)
            p2 = (nodes[b].pos * scale + offset).astype(int)
            cv2.line(frame, tuple(p1), tuple(p2), (255, 255, 255), 1)
        
        for i, n in enumerate(nodes):
            pos = (n.pos * scale + offset).astype(int)
            col = (0, 255, 140) if i in important else (255, 255, 255)
            cv2.circle(frame, tuple(pos), 5, col, -1)
            cv2.circle(frame, tuple(pos), 9, (0, 245, 255), 1)
            
            tx = pos[0] + 22
            ty = pos[1] - 11
            (tw, th), _ = cv2.getTextSize(n.name, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            cv2.rectangle(frame, (tx-7, ty-th-5), (tx+tw+8, ty+6), (0,245,255), 1)
            cv2.putText(frame, n.name, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.52, col, 1)
        
        if highlighted is not None:
            pos = (nodes[highlighted].pos * scale + offset).astype(int)
            cv2.circle(frame, tuple(pos), 27, (0, 255, 255), 2)
        
        # Expanded AI 2027 panel
        if expanded_node is not None:
            px = w - 530
            py = 140
            cv2.rectangle(frame, (px, py), (px + 490, py + 380), (0, 25, 45), -1)
            cv2.rectangle(frame, (px, py), (px + 490, py + 380), (0, 245, 255), 3)
            
            title = nodes[expanded_node].name
            cv2.putText(frame, title.upper(), (px + 35, py + 55), cv2.FONT_HERSHEY_DUPLEX, 1.18, (0, 255, 140), 3)
            
            lines = details_dict.get(expanded_node, ["Loading from AI 2027..."])
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (px + 38, py + 115 + i * 42), cv2.FONT_HERSHEY_DUPLEX, 0.68, (220, 230, 255), 1)
            
            cv2.putText(frame, "FIST to close", (px + 38, py + 355), cv2.FONT_HERSHEY_DUPLEX, 0.58, (100, 100, 255), 1)
        
        # Tesseract
        draw_tesseract(frame, w - 170, 170, 118, anim_time)
        cv2.putText(frame, "TESSERACT", (w - 225, 310), cv2.FONT_HERSHEY_DUPLEX, 0.82, (0, 245, 255), 2)
        
        # Header
        cv2.putText(frame, "TOPOLOGIES OF THOUGHTS", (w//2 - 270, 68), cv2.FONT_HERSHEY_DUPLEX, 1.38, (0, 245, 255), 3)
        cv2.putText(frame, "mode: two-hand stretch expansion • directional swipe", (w//2 - 290, 98), cv2.FONT_HERSHEY_DUPLEX, 0.64, (180, 180, 180), 1)
        
        # Status
        status = ""
        if gesture == "fist": status = "FROZEN ✊"
        elif len(hand_list) == 2 and hand_list[0][1] == "pinch" and hand_list[1][1] == "pinch": status = "STRETCH APART TO EXPAND"
        elif gesture == "pinch": status = "PINCH + SWIPE → NEXT THOUGHT"
        elif gesture == "point": status = "ZOOM IN"
        if status:
            cv2.putText(frame, status, (45, h - 45), cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 255, 255), 2)
        
        cv2.imshow("AI 2027 • Two-Hand Neural Web", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()