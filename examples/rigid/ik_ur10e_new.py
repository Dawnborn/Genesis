import argparse

import numpy as np

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug", backend=gs.cuda) # by default: backend=gs.cpu
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=False,
            enable_self_collision=False,
            # gravity=(0, 0, -9.8),
            gravity=(0,0,0)
        ),
        # renderer=gs.renderers.RayTracer(  # type: ignore
        #     env_surface=gs.surfaces.Emission(
        #         emissive_texture=gs.textures.ImageTexture(
        #             image_path="textures/indoor_bright.png",
        #         ),
        #     ),
        #     env_radius=15.0,
        #     env_euler=(0, 0, 180),
        #     lights=[
        #         {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
        #     ],
        # ),
    )

    ########################## entities ##########################
    # floor
    plane = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, -0.5),
        ),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/ur10e/robot_wsg50.urdf",fixed=True),
    )

    joint_names = [ # robot.joints
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "finger_left_joint",
        "finger_right_joint"
    ]

    # asset's own attributes
    sphere = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -2.3, 0.0),
        ),
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## sensors ##########################

    camera_side = scene.add_camera(
        res=(640, 480),
        pos=(8.5, 0.0, 1.5),
        lookat=(3.0, 0.0, 0.7),
        fov=60,
        GUI=True,
        spp=512,
    )

    ########################## build ##########################
    scene.build()

    ### forward
    scene.reset()

    # dofs_idx = [robot.get_joint(name).dof_idx_local for name in joint_names]
    # robot.set_dofs_kp(
    #     kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000]),
    #     dofs_idx_local = dofs_idx,
    # )
    # # set velocity gains
    # robot.set_dofs_kv(
    #     kv             = np.array([450, 450, 350, 350, 200, 200]),
    #     dofs_idx_local = dofs_idx,
    # )
    # # set force range for safety
    # robot.set_dofs_force_range(
    #     lower          = np.array([-87, -87, -87, -87, -12, -12]),
    #     upper          = np.array([ 87,  87,  87,  87,  12,  12]),
    #     dofs_idx_local = dofs_idx,
    # )

    target_quat = np.array([0, 1, 0, 0])  # pointing downwards
    target_quat = np.array([1, 0, 0, 0])
    center = np.array([0.4, -0.2, 0.25])
    r = 0.1

    ee_link = robot.get_link("wrist_3_link")
    print(ee_link.get_pos())
    print(ee_link.get_quat())

    # for i in range(0, 2000):
    i = 0
    while(True):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r

        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q, err = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            return_error=True,
            rot_mask=[True, True, True],  # for demo purpose: only care about direction of z-axis
        )
        print("error:", err)

        # Note that this IK example is only for visualizing the solved q, so here we do not call scene.step(), but only update the state and the visualizer
        # In actual control applications, you should instead use robot.control_dofs_position() and scene.step()
        
        # q[-2:] = 0
        q[-1] = 0.02
        q[-2] = 0.02
        # robot.set_qpos(q)
        robot.control_dofs_position(q)
        scene.step()
        scene.visualizer.update()
        camera_side.render()



if __name__ == "__main__":
    main()
