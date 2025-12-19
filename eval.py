import cv2
import cnn
import torch
import torch.nn.functional as F

model = cnn.SimpleCNN()
model.load_state_dict(torch.load("simple_cnn_weights.pth", weights_only=True))
model.to(cnn.device)
model.eval()

#cap = cv2.VideoCapture(0)

eval_set_path = "C:\\Python Projects\\BlenderProc\\eval_set\\"

for i in range(40):
    frame = cv2.imread(eval_set_path + str(i) + ".png")
    #ret, frame = cap.read()
    #frame = cv2.resize(frame, (128,128))
    # Run model
    image_tensor = cnn.image_to_tensor(frame)
    image_tensor = image_tensor.to(cnn.device)
    output = model(image_tensor)
    pred_class = torch.argmax(output, dim=1).item()  # dim=1 to get class per sample
    probs = F.softmax(output, dim=1)
    pred_prob = probs[0, pred_class].item()

    # Output results
    text = "NOTHING"
    if(pred_class == 1):
        text = "STOP"
    elif(pred_class == 2):
        text = "PRIORITY"
    elif(pred_class == 3):
        text = "ROUNDABOUT"

    frame = cv2.resize(frame, (0, 0), fx=5, fy=5)

    cv2.putText(
        frame,
        text,  # text
        (5, 50),  # bottom-left corner (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        2,  # font scale
        (0, 255, 0),  # color (B, G, R)
        2,  # thickness
        cv2.LINE_AA  # line type (anti-aliased)
    )

    cv2.putText(
        frame,
        format(pred_prob, ".2f"),  # text
        (5, 100),  # bottom-left corner (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        2,  # font scale
        (0, 255, 0),  # color (B, G, R)
        2,  # thickness
        cv2.LINE_AA  # line type (anti-aliased)
    )

    cv2.imshow("Video Frame", frame)

        # Press 'q' to exit
    if cv2.waitKey(0) == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()