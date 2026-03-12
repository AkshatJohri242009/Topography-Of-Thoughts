"""
Topologies of Thoughts
A hand-tracked, live knowledge graph explorer.
Runs at 60fps with MediaPipe hand tracking overlaid on webcam feed.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
from collections import defaultdict

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
WINDOW_NAME = "Topologies of Thoughts"
TARGET_FPS = 60

# Colors (BGR)
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)
GREEN       = (80, 220, 100)
GREEN_DIM   = (40, 120, 55)
CYAN        = (200, 240, 220)
ORANGE_RED  = (50, 80, 230)
GRAY        = (160, 160, 160)
GRAY_DIM    = (80, 80, 80)
PANEL_BG    = (18, 18, 18)

# Graph physics
SPRING_REST      = 120
SPRING_K         = 0.012
REPULSION        = 6000
DAMPING          = 0.82
CENTER_PULL      = 0.003
CLUSTER_PULL     = 0.006

# Node interaction
HOVER_RADIUS     = 55
DRAG_SNAP_DIST   = 40
PINCH_THRESH     = 0.07

# ─── KNOWLEDGE GRAPH DATA ────────────────────────────────────────────────────────
# Nodes: (label, cluster_id, sub_notes)
RAW_NODES = [
    # Cluster 0: Generative & Creative AI
    ("Generative Media\n& Notation",         0, ["notation systems", "AR/VR output", "multimodal gen"]),
    ("Latent Space\nNavigation",             0, ["embedding topology", "semantic walk", "interpolation"]),
    ("Procedural\nAesthetics",               0, ["generative art", "rule-based beauty", "emergence"]),
    ("LLM Prompting\nCraft",                 0, ["chain-of-thought", "few-shot", "prompt topology"]),
    ("Diffusion Model\nArchitecture",        0, ["U-Net", "DDPM", "score matching"]),

    # Cluster 1: Mind, Language, Cognition
    ("Cognition,\nLanguage & AI",            1, ["embodied cognition", "Sapir-Whorf", "LLM thought"]),
    ("History of\nInformation Tools",        1, ["Turing", "Memex", "Engelbart", "hypertext"]),
    ("Distributed\nCognition",               1, ["extended mind", "Clark & Chalmers", "tools as cognition"]),
    ("Semantic\nMemory Structures",          1, ["graph memory", "associative nets", "schema theory"]),
    ("Linguistics &\nSymbol Grounding",      1, ["grounding problem", "syntax trees", "pragmatics"]),

    # Cluster 2: Interaction & Interface
    ("Spatial\nInteraction Design",          2, ["gesture UI", "3D interfaces", "embodied interaction"]),
    ("Hand Gesture\nVocabularies",           2, ["pinch", "swipe", "dwell", "bimanual ops"]),
    ("HCI Paradigms",                        2, ["WIMP", "NUI", "post-WIMP", "tangible UI"]),
    ("AR Overlay\nSystems",                  2, ["registration", "occlusion", "spatial audio"]),

    # Cluster 3: Knowledge Graphs & PKM
    ("ArXFusion: Lore\n& Mythologies",       3, ["world-building", "narrative graphs", "canon nets"]),
    ("PKM\nMethodologies",                   3, ["Zettelkasten", "PARA", "linking thoughts"]),
    ("Ontology\nDesign",                     3, ["OWL", "RDF", "taxonomies", "knowledge graphs"]),
    ("Note Graph\nTopologies",               3, ["hubs", "clusters", "bridges", "isolated nodes"]),

    # Cluster 4: Systems & Emergence
    ("Network\nTopology Theory",             4, ["scale-free", "small-world", "Erdős-Rényi"]),
    ("Complex\nAdaptive Systems",            4, ["emergence", "self-org", "attractors"]),
    ("Game Design\n& Semiotics",             4, ["ludonarrative", "sign systems", "mechanics as meaning"]),
    ("Social Design\n& Aesthetics",          4, ["participatory design", "co-creation", "beauty norms"]),
]

# Edges: (node_index_a, node_index_b, relationship_label)
RAW_EDGES = [
    (0,  1,  "spaces ideas live in"),
    (0,  2,  "aesthetic output"),
    (1,  3,  "navigation via prompts"),
    (3,  4,  "prompting diffusion"),
    (5,  6,  "tools shape thought"),
    (5,  7,  "extends cognition"),
    (7,  8,  "memory as graph"),
    (8,  9,  "symbols ground meaning"),
    (10, 11, "gestures are vocabulary"),
    (10, 13, "gesture enables AR"),
    (12, 10, "paradigm evolution"),
    (14, 15, "mythology as PKM"),
    (15, 16, "PKM uses ontologies"),
    (16, 17, "ontology IS topology"),
    (18, 19, "topology → emergence"),
    (20, 21, "games as social design"),
    # Cross-cluster
    (0,  5,  "media shapes cognition"),
    (5,  15, "thinking as note-taking"),
    (10, 7,  "interface extends mind"),
    (17, 18, "note graph = network"),
    (3,  5,  "LLMs and language"),
    (1,  17, "latent = thought topology"),
    (19, 21, "systems → aesthetics"),
    (8,  16, "memory as ontology"),
    (11, 10, "gesture ↔ interaction"),
    (6,  7,  "history of ext. mind"),
    (4,  3,  "diffusion navigates latent"),
    (2,  20, "procedural game aesthetics"),
    (13, 11, "AR needs gestures"),
    (14, 19, "lore = complex system"),
]

TOPOLOGY_MODES = [
    {
        "name": "decentralized",
        "description": "notes in different\nclusters by topic",
        "subtitle": "decentralized: notes cluster into themed hubs",
    },
    {
        "name": "distributed",
        "description": "notes connected by\nrelationships via edges",
        "subtitle": "distributed: edges labeled by llm describing how ideas connect",
    },
    {
        "name": "centralized",
        "description": "one central thought\nconnected to all other nodes",
        "subtitle": "centralized: one core idea connecting all",
    },
]

# ─── MINI DIAGRAM SHAPES ─────────────────────────────────────────────────────────
def draw_mini_diagram(frame, cx, cy, mode_idx, size=38):
    """Draw the small topology icon matching each mode."""
    if mode_idx == 0:  # decentralized — clusters
        clusters = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        sub = [(-0.3, 0), (0, -0.3), (0.3, 0)]
        for bx, by in clusters:
            for sx, sy in sub:
                px = int(cx + (bx * 0.55 + sx * 0.35) * size)
                py = int(cy + (by * 0.55 + sy * 0.35) * size)
                cv2.circle(frame, (px, py), 2, WHITE, -1)
            # dot for center of cluster
            cv2.circle(frame, (int(cx + bx * 0.55 * size), int(cy + by * 0.55 * size)), 3, WHITE, -1)
        # intra-cluster lines
        for bx, by in clusters:
            for sx, sy in sub:
                px = int(cx + (bx * 0.55 + sx * 0.35) * size)
                py = int(cy + (by * 0.55 + sy * 0.35) * size)
                cv2.line(frame,
                    (int(cx + bx*0.55*size), int(cy + by*0.55*size)),
                    (px, py), WHITE, 1, cv2.LINE_AA)

    elif mode_idx == 1:  # distributed — mesh
        pts = []
        for i in range(8):
            a = i / 8 * 2 * math.pi + 0.3
            r = size * (0.55 + 0.3 * (i % 2))
            pts.append((int(cx + math.cos(a)*r), int(cy + math.sin(a)*r)))
        connections = [(0,3),(1,5),(2,6),(3,7),(0,4),(2,5),(4,6),(1,7),(0,2)]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], WHITE, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 2, WHITE, -1)

    elif mode_idx == 2:  # centralized — star
        center = (cx, cy)
        cv2.circle(frame, center, 4, WHITE, -1)
        for i in range(10):
            a = i / 10 * 2 * math.pi
            ex = int(cx + math.cos(a) * size * 0.85)
            ey = int(cy + math.sin(a) * size * 0.85)
            cv2.line(frame, center, (ex, ey), WHITE, 1, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 2, WHITE, -1)


# ─── NODE CLASS ──────────────────────────────────────────────────────────────────
class Node:
    def __init__(self, idx, label, cluster_id, sub_notes):
        self.idx        = idx
        self.label      = label
        self.cluster_id = cluster_id
        self.sub_notes  = sub_notes
        self.x          = 0.0
        self.y          = 0.0
        self.vx         = 0.0
        self.vy         = 0.0
        self.ax         = 0.0
        self.ay         = 0.0
        self.hovered    = False
        self.dragged    = False
        self.alpha      = 1.0      # fade in

CLUSTER_COLORS = [
    (80,  220, 100),   # green
    (220, 180, 80),    # gold
    (100, 180, 255),   # blue
    (240, 120, 80),    # orange
    (180, 100, 240),   # purple
]


# ─── GRAPH PHYSICS ───────────────────────────────────────────────────────────────
class GraphPhysics:
    def __init__(self, nodes, edges, mode_idx, w, h):
        self.nodes    = nodes
        self.edges    = edges
        self.mode_idx = mode_idx
        self.w        = w
        self.h        = h
        self._init_positions()

    def _init_positions(self):
        """Scatter nodes with some cluster grouping."""
        cluster_centers = {}
        num_clusters = max(n.cluster_id for n in self.nodes) + 1
        for c in range(num_clusters):
            a = c / num_clusters * 2 * math.pi
            cx = self.w * 0.35 + math.cos(a) * self.w * 0.22
            cy = self.h * 0.5  + math.sin(a) * self.h * 0.28
            cluster_centers[c] = (cx, cy)
        for n in self.nodes:
            cx, cy = cluster_centers[n.cluster_id]
            n.x = cx + random.uniform(-60, 60)
            n.y = cy + random.uniform(-60, 60)
            n.vx = n.vy = 0.0

    def set_mode(self, mode_idx):
        self.mode_idx = mode_idx

    def step(self, dt):
        n = self.nodes
        # Reset forces
        for node in n:
            node.ax = 0.0
            node.ay = 0.0

        # Repulsion
        for i in range(len(n)):
            for j in range(i+1, len(n)):
                dx = n[i].x - n[j].x
                dy = n[i].y - n[j].y
                d2 = dx*dx + dy*dy + 1e-4
                d  = math.sqrt(d2)
                f  = REPULSION / d2
                fx = f * dx / d
                fy = f * dy / d
                n[i].ax += fx; n[i].ay += fy
                n[j].ax -= fx; n[j].ay -= fy

        # Spring attraction along edges
        for a_idx, b_idx, _ in self.edges:
            na, nb = n[a_idx], n[b_idx]
            dx = nb.x - na.x
            dy = nb.y - na.y
            d  = math.sqrt(dx*dx + dy*dy) + 1e-4
            rest = SPRING_REST
            if self.mode_idx == 2:   # centralized — tighter to center
                rest = SPRING_REST * 0.6
            f  = SPRING_K * (d - rest)
            fx = f * dx / d
            fy = f * dy / d
            na.ax += fx; na.ay += fy
            nb.ax -= fx; nb.ay -= fy

        # Mode-specific forces
        cx = self.w * 0.38
        cy = self.h * 0.5

        if self.mode_idx == 0:  # decentralized: pull to cluster centers
            num_clusters = max(nd.cluster_id for nd in n) + 1
            for c in range(num_clusters):
                a = c / num_clusters * 2 * math.pi
                tcx = cx + math.cos(a) * self.w * 0.22
                tcy = cy + math.sin(a) * self.h * 0.28
                for nd in n:
                    if nd.cluster_id == c:
                        nd.ax += CLUSTER_PULL * (tcx - nd.x)
                        nd.ay += CLUSTER_PULL * (tcy - nd.y)

        elif self.mode_idx == 1:  # distributed: gentle centering only
            for nd in n:
                nd.ax += CENTER_PULL * (cx - nd.x)
                nd.ay += CENTER_PULL * (cy - nd.y)

        elif self.mode_idx == 2:  # centralized: node 0 is center, rest orbit
            central = n[0]
            central.ax += 0.08 * (cx - central.x)
            central.ay += 0.08 * (cy - central.y)
            for nd in n[1:]:
                nd.ax += CENTER_PULL * 0.5 * (cx - nd.x)
                nd.ay += CENTER_PULL * 0.5 * (cy - nd.y)

        # Integrate
        for nd in n:
            if nd.dragged:
                nd.vx = nd.vy = 0.0
                continue
            nd.vx = (nd.vx + nd.ax * dt) * DAMPING
            nd.vy = (nd.vy + nd.ay * dt) * DAMPING
            nd.x += nd.vx * dt
            nd.y += nd.vy * dt
            # Boundary
            margin = 60
            nd.x = max(margin, min(self.w * 0.72 - margin, nd.x))
            nd.y = max(margin, min(self.h - margin, nd.y))


# ─── RENDERER ────────────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self._font     = cv2.FONT_HERSHEY_SIMPLEX
        self._mono     = cv2.FONT_HERSHEY_PLAIN

    def draw_edge(self, frame, a, b, label, mode_idx, hovered_pair=False):
        x1, y1 = int(a.x), int(a.y)
        x2, y2 = int(b.x), int(b.y)
        alpha = 0.18 if not hovered_pair else 0.7
        color = (int(255*alpha),) * 3

        overlay = frame.copy()
        thick = 2 if hovered_pair else 1
        cv2.line(overlay, (x1,y1), (x2,y2), WHITE, thick, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        # Edge label in distributed mode
        if mode_idx == 1 and label and (a.hovered or b.hovered):
            mx, my = (x1+x2)//2, (y1+y2)//2
            words = label.split()
            lw = max(len(w) for w in words) if words else 4
            tw = lw * 6 + 8
            th = len(words) * 12 + 6
            cv2.rectangle(frame, (mx-tw//2-2, my-th//2-2), (mx+tw//2+2, my+th//2+2),
                          (20,20,20), -1)
            for wi, w in enumerate(words):
                cv2.putText(frame, w,
                    (mx - lw*3, my - (len(words)-1)*6 + wi*12),
                    self._mono, 0.55, ORANGE_RED, 1, cv2.LINE_AA)

    def draw_node(self, frame, nd, mode_idx):
        x, y = int(nd.x), int(nd.y)
        color = CLUSTER_COLORS[nd.cluster_id % len(CLUSTER_COLORS)]
        lines = nd.label.split('\n')

        font_scale = 0.38
        line_h = 14
        max_w = max(cv2.getTextSize(l, self._font, font_scale, 1)[0][0] for l in lines)
        total_h = len(lines) * line_h

        if nd.hovered or nd.dragged:
            # Bright green bracket style
            pad = 5
            bx1 = x - max_w//2 - pad
            by1 = y - total_h//2 - pad
            bx2 = x + max_w//2 + pad
            by2 = y + total_h//2 + pad + 2
            blen = 10  # bracket arm length
            bt   = 2   # bracket thickness

            # Top-left bracket
            cv2.line(frame, (bx1, by1+blen), (bx1, by1), GREEN, bt, cv2.LINE_AA)
            cv2.line(frame, (bx1, by1), (bx1+blen, by1), GREEN, bt, cv2.LINE_AA)
            # Top-right bracket
            cv2.line(frame, (bx2-blen, by1), (bx2, by1), GREEN, bt, cv2.LINE_AA)
            cv2.line(frame, (bx2, by1), (bx2, by1+blen), GREEN, bt, cv2.LINE_AA)
            # Bottom-left bracket
            cv2.line(frame, (bx1, by2-blen), (bx1, by2), GREEN, bt, cv2.LINE_AA)
            cv2.line(frame, (bx1, by2), (bx1+blen, by2), GREEN, bt, cv2.LINE_AA)
            # Bottom-right bracket
            cv2.line(frame, (bx2-blen, by2), (bx2, by2), GREEN, bt, cv2.LINE_AA)
            cv2.line(frame, (bx2, by2-blen), (bx2, by2), GREEN, bt, cv2.LINE_AA)

            for li, line in enumerate(lines):
                ty = y - total_h//2 + li*line_h + line_h - 2
                cv2.putText(frame, line, (x - max_w//2, ty),
                            self._font, font_scale, GREEN, 1, cv2.LINE_AA)

            # Sub-notes below
            if nd.hovered:
                sny = by2 + 6
                for si, sn in enumerate(nd.sub_notes[:3]):
                    cv2.putText(frame, f"· {sn}",
                                (bx1, sny + si*11),
                                self._mono, 0.45, GREEN_DIM, 1, cv2.LINE_AA)
        else:
            # Dim dot + tiny white text
            cv2.circle(frame, (x, y), 2,
                       tuple(int(c*0.7) for c in color), -1)
            for li, line in enumerate(lines):
                ty = y - total_h//2 + li*line_h + line_h - 2
                tx = x - max_w//2
                # shadow
                cv2.putText(frame, line, (tx+1, ty+1),
                            self._font, font_scale, (20,20,20), 1, cv2.LINE_AA)
                cv2.putText(frame, line, (tx, ty),
                            self._font, font_scale,
                            tuple(int(c*0.85) for c in color), 1, cv2.LINE_AA)

    def draw_graph(self, frame, mode_idx, hovered_node_idx):
        # Edges first
        for ai, bi, lbl in self.edges:
            na, nb = self.nodes[ai], self.nodes[bi]
            hp = (na.hovered or nb.hovered)
            self.draw_edge(frame, na, nb, lbl, mode_idx, hp)
        # Nodes on top
        for nd in self.nodes:
            self.draw_node(frame, nd, mode_idx)


# ─── PANEL RENDERER ──────────────────────────────────────────────────────────────
def draw_panel(frame, mode_idx, h, w):
    """Draw the top-right 'Topologies of Thoughts' info panel."""
    pw  = 190
    ph  = 185
    px  = w - pw - 12
    py  = 12
    pad = 14

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (14,14,14), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Topologies of",
                (px+pad, py+22), font, 0.42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Thoughts",
                (px+pad, py+38), font, 0.42, WHITE, 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (px+pad, py+46), (px+pw-pad, py+46), GRAY_DIM, 1)

    # Mini diagram
    draw_mini_diagram(frame, px + pw//2, py + 100, mode_idx, size=34)

    # Mode name
    mode = TOPOLOGY_MODES[mode_idx]
    cv2.putText(frame, f"mode: {mode['name']}",
                (px+pad, py+148), font, 0.32, GRAY, 1, cv2.LINE_AA)
    # Description
    for li, dl in enumerate(mode['description'].split('\n')):
        cv2.putText(frame, dl,
                    (px+pad, py+162 + li*13), font, 0.30, GRAY_DIM, 1, cv2.LINE_AA)


def draw_subtitle(frame, mode_idx, h, w):
    """Bottom subtitle text like in the video."""
    mode = TOPOLOGY_MODES[mode_idx]
    lines = mode['subtitle'].split('\n') if '\n' in mode['subtitle'] else [mode['subtitle']]
    # Wrap long line
    all_lines = []
    for line in lines:
        words = line.split()
        cur = ""
        for word in words:
            if len(cur) + len(word) + 1 > 38:
                all_lines.append(cur.strip())
                cur = word + " "
            else:
                cur += word + " "
        if cur.strip():
            all_lines.append(cur.strip())

    font = cv2.FONT_HERSHEY_SIMPLEX
    for li, l in enumerate(all_lines):
        y = h - 18 - (len(all_lines)-1-li) * 20
        cv2.putText(frame, l, (22, y+1), font, 0.52, BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, l, (22, y),   font, 0.52, WHITE, 1, cv2.LINE_AA)


def draw_hand_cursor(frame, hx, hy, pinching):
    """Draw hand cursor at index fingertip."""
    if pinching:
        cv2.circle(frame, (hx, hy), 10, GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, (hx, hy), 3,  GREEN, -1)
    else:
        cv2.circle(frame, (hx, hy), 6, WHITE, 1, cv2.LINE_AA)
        cv2.circle(frame, (hx, hy), 2, WHITE, -1)


def draw_corner_brackets(frame, h, w):
    """Draw subtle L-shaped corner brackets around the camera frame area."""
    arm = 28
    t   = 2
    col = (130, 130, 130)
    margin = 8
    # We draw on the whole frame
    # Top-left
    cv2.line(frame, (margin, margin+arm), (margin, margin), col, t, cv2.LINE_AA)
    cv2.line(frame, (margin, margin), (margin+arm, margin), col, t, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (w-margin-arm, margin), (w-margin, margin), col, t, cv2.LINE_AA)
    cv2.line(frame, (w-margin, margin), (w-margin, margin+arm), col, t, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (margin, h-margin-arm), (margin, h-margin), col, t, cv2.LINE_AA)
    cv2.line(frame, (margin, h-margin), (margin+arm, h-margin), col, t, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (w-margin-arm, h-margin), (w-margin, h-margin), col, t, cv2.LINE_AA)
    cv2.line(frame, (w-margin, h-margin-arm), (w-margin, h-margin), col, t, cv2.LINE_AA)


def draw_fps(frame, fps):
    cv2.putText(frame, f"{fps:.0f} fps",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90,90,90), 1, cv2.LINE_AA)


def draw_mode_hint(frame, h, w):
    """Small key hint."""
    cv2.putText(frame, "[ 1/2/3 ] mode   [ R ] reset   [ Q ] quit",
                (w//2 - 170, h - 6),
                cv2.FONT_HERSHEY_PLAIN, 0.88, (60,60,60), 1, cv2.LINE_AA)


# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    # Build nodes & edges
    nodes = [Node(i, lbl, cid, sn) for i, (lbl, cid, sn) in enumerate(RAW_NODES)]
    edges = RAW_EDGES  # (a_idx, b_idx, label)

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    ret, frame = cap.read()
    if not ret:
        print("Cannot open camera")
        return
    h, w = frame.shape[:2]

    physics  = GraphPhysics(nodes, edges, 0, w, h)
    renderer = Renderer(nodes, edges)

    # MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        model_complexity=1,
    )

    mode_idx       = 0
    dragged_node   = None
    hand_x, hand_y = -1, -1
    pinching       = False

    prev_time      = time.time()
    fps_display    = 60.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    print("=== Topologies of Thoughts ===")
    print("Keys: 1/2/3 = switch mode | R = reset layout | Q/Esc = quit")

    frame_skip = 0  # process hand every frame for responsiveness

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror
        frame = cv2.flip(frame, 1)

        # ── Hand Tracking ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        pinching = False
        hovered_node_idx = -1

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            # Index fingertip = landmark 8
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)
            hand_x, hand_y = ix, iy

            # Thumb tip = landmark 4
            tx = int(lm[4].x * w)
            ty = int(lm[4].y * h)

            # Pinch detection: normalized distance thumb↔index
            pdx = lm[4].x - lm[8].x
            pdy = lm[4].y - lm[8].y
            pinch_dist = math.sqrt(pdx*pdx + pdy*pdy)
            pinching = pinch_dist < PINCH_THRESH

            # Hover detection
            best_d = HOVER_RADIUS
            hovered_node_idx = -1
            for nd in nodes:
                d = math.sqrt((nd.x - hand_x)**2 + (nd.y - hand_y)**2)
                if d < best_d:
                    best_d = d
                    hovered_node_idx = nd.idx

            for nd in nodes:
                nd.hovered = (nd.idx == hovered_node_idx)

            # Drag logic
            if pinching:
                if dragged_node is None and hovered_node_idx >= 0:
                    dragged_node = nodes[hovered_node_idx]
                if dragged_node is not None:
                    dragged_node.x  = float(hand_x)
                    dragged_node.y  = float(hand_y)
                    dragged_node.vx = 0.0
                    dragged_node.vy = 0.0
                    dragged_node.dragged  = True
                    dragged_node.hovered  = True
            else:
                if dragged_node is not None:
                    dragged_node.dragged = False
                dragged_node = None
        else:
            hand_x = hand_y = -1
            for nd in nodes:
                nd.hovered = False
            if dragged_node:
                dragged_node.dragged = False
            dragged_node = None

        # ── Physics step ──
        dt = 16.0  # ~60fps step in ms
        physics.step(dt)

        # ── Render ──
        # 1. Graph (edges + nodes) on clean frame
        renderer.draw_graph(frame, mode_idx, hovered_node_idx)

        # 2. Corner brackets
        draw_corner_brackets(frame, h, w)

        # 3. Right panel
        draw_panel(frame, mode_idx, h, w)

        # 4. Subtitle
        draw_subtitle(frame, mode_idx, h, w)

        # 5. Hand cursor
        if hand_x > 0:
            draw_hand_cursor(frame, hand_x, hand_y, pinching)

        # 6. FPS counter
        now = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - prev_time, 1e-5))
        prev_time = now
        draw_fps(frame, fps_display)

        # 7. Key hint
        draw_mode_hint(frame, h, w)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == ord('1'):
            mode_idx = 0
            physics.set_mode(0)
        elif key == ord('2'):
            mode_idx = 1
            physics.set_mode(1)
        elif key == ord('3'):
            mode_idx = 2
            physics.set_mode(2)
        elif key in (ord('r'), ord('R')):
            physics._init_positions()
            for nd in nodes:
                nd.vx = nd.vy = 0.0

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()