import cv2
import pandas as pd


IMAGE_PATH = r"F:\python\Task_1\pic2.jpg"
CSV_PATH = r"F:\python\Task_1\colors.csv"

# -------------------- READ CSV --------------------
columns = ["color", "color_name", "hex", "R", "G", "B"]
colors = pd.read_csv(CSV_PATH, names=columns)

# -------------------- READ IMAGE --------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("❌ Error: Image not found. Check IMAGE_PATH")
    exit()

img = cv2.resize(img, (800, 600))

# -------------------- GLOBAL VARIABLES --------------------
clicked = False
r = g = b = x_pos = y_pos = 0

# -------------------- COLOR MATCH FUNCTION --------------------
def get_color_name(R, G, B):
    min_dist = 10000
    cname = ""

    for i in range(len(colors)):
        d = abs(R - colors.loc[i, "R"]) + \
            abs(G - colors.loc[i, "G"]) + \
            abs(B - colors.loc[i, "B"])

        if d < min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]

    return cname

# -------------------- MOUSE CALLBACK FUNCTION --------------------
def mouse_callback(event, x, y, flags, param):
    global clicked, r, g, b, x_pos, y_pos

    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = img[y, x]
        b, g, r = int(b), int(g), int(r)

# -------------------- WINDOW SETUP --------------------
cv2.namedWindow("Color Detection")
cv2.setMouseCallback("Color Detection", mouse_callback)

# -------------------- MAIN LOOP --------------------
while True:
    display_img = img.copy()

    if clicked:
        # Color display rectangle
        cv2.rectangle(display_img, (20, 20), (650, 60), (b, g, r), -1)

        text = f"{get_color_name(r, g, b)}  R={r} G={g} B={b}"

        # Text color based on brightness
        text_color = (255, 255, 255)
        if r + g + b > 600:
            text_color = (0, 0, 0)

        cv2.putText(
            display_img,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2
        )

        # Mark clicked point
        cv2.circle(display_img, (x_pos, y_pos), 10, (0, 255, 255), -1)

    cv2.imshow("Color Detection", display_img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()