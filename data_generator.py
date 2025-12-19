# Imports
import blenderproc as bproc
import numpy as np
import cv2
import random
import json

# Necessary at the beginning of every BlenderProc script
bproc.init()

# Path for Haven backgrounds we want to use
hdri_dir = "C:\\Python Projects\\BlenderProc\\haven" # Replace with yours

# Path for traffic sign objs
obj_dir ="C:\\Python Projects\\BlenderProc\\objs" # Replace with yours

# Path for generated images

generated_path = ""

# Loading the traffic sign objs
stop = bproc.loader.load_obj(obj_dir+"\\stop.obj")[0]
priority = bproc.loader.load_obj(obj_dir+"\\priority.obj")[0]
roundabout = bproc.loader.load_obj(obj_dir+"\\roundabout.obj")[0]

# Store signs in list
signs = [stop, priority, roundabout]

# Setting custom properties for segmentation output
stop.set_cp("id", "stop")
priority.set_cp("id", "priority")
roundabout.set_cp("id", "roundabout")


# Setting up the camera
bproc.camera.set_intrinsics_from_blender_params(
    lens=34,
    lens_unit="MILLIMETERS",
    clip_start=0,
    clip_end=100,
    image_width=128,
    image_height=128
)
# Main rendering loop
for i in range(0,40):
    j = random.choice([0, 1])
    # Forget previous camera poses
    bproc.utility.reset_keyframes()

    if j == 0:
        chosen = random.choice(signs)


        chosen.set_location([random.uniform(-3, 3),
                             0,
                             random.uniform(-3, 3)])

        chosen.set_rotation_euler([np.pi / 2,
                                   random.uniform(-np.pi/4, np.pi/4),
                                   random.uniform(-np.pi/4, np.pi/4)])

        scale = random.uniform(2, 5)
        chosen.set_scale([scale, scale, scale])


        for sign in signs:
            if sign is not chosen:
                sign.set_location([100, 0, 0])

        data = {
            "name": chosen.get_cp("id"),
        }

    else:
        for sign in signs:
            sign.set_location([100, 0, 0])
        data = {
            "name": "nothing",
        }

    with open(generated_path+str(i)+".json", "w") as f:
        json.dump(data, f, indent=4)


    # Load a random Haven background
    hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(hdri_dir)
    bproc.world.set_world_background_hdr_img(hdri_path, strength=1.0, rotation_euler=None)

    # Set the camera to be in front of the object
    cam_pose = bproc.math.build_transformation_mat([0, -10, 0], [np.pi / 2, 0, 0])
    bproc.camera.add_camera_pose(cam_pose)

    # Render the scene
    data = bproc.renderer.render()

    # Convert color image to OpenCV format
    color_image = data['colors'][0]
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0, 1)
    color_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blur = random.choice([1, 3, 5])
    color_image = cv2.GaussianBlur(color_image, (blur, blur), 0)

    # Visualize results
    cv2.imshow("image", color_image)
    cv2.waitKey(1)

    # Save results to disk

    cv2.imwrite(generated_path + str(i) + ".png", color_image)
