dataset_info = dict(
    dataset_name="llvip",
    paper_info=dict(
        author="{Cormier, Mickael and Ng Zhi Yi, Caleb and Specker, Andreas and Bla{\ss}, Benjamin and Heizmann, Michael and Beyerer, J{\"u}rgen",
        title="Leveraging Thermal Imaging for Robust Human Pose Estimation in Low-Light Vision",
        container="Proceedings of the Asian Conference on Computer Vision (ACCV) Workshops",
        year="2024",
        homepage="https://github.com/MickaelCormier/llvip-pose",
    ),
    keypoint_info={
        0: dict(name="nose", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(name="head_bottom", id=1, color=[51, 153, 255], type="upper", swap=""),
        2: dict(name="head_top", id=2, color=[51, 153, 255], type="upper", swap=""),
        3: dict(name="left_ear", id=3, color=[51, 153, 255], type="upper", swap="right_ear"),
        4: dict(name="right_ear", id=4, color=[51, 153, 255], type="upper", swap="left_ear"),
        5: dict(name="left_shoulder", id=5, color=[0, 255, 0], type="upper", swap="right_shoulder"),
        6: dict(name="right_shoulder", id=6, color=[255, 128, 0], type="upper", swap="left_shoulder"),
        7: dict(name="left_elbow", id=7, color=[0, 255, 0], type="upper", swap="right_elbow"),
        8: dict(name="right_elbow", id=8, color=[255, 128, 0], type="upper", swap="left_elbow"),
        9: dict(name="left_wrist", id=9, color=[0, 255, 0], type="upper", swap="right_wrist"),
        10: dict(name="right_wrist", id=10, color=[255, 128, 0], type="upper", swap="left_wrist"),
        11: dict(name="left_hip", id=11, color=[0, 255, 0], type="lower", swap="right_hip"),
        12: dict(name="right_hip", id=12, color=[255, 128, 0], type="lower", swap="left_hip"),
        13: dict(name="left_knee", id=13, color=[0, 255, 0], type="lower", swap="right_knee"),
        14: dict(name="right_knee", id=14, color=[255, 128, 0], type="lower", swap="left_knee"),
        15: dict(name="left_ankle", id=15, color=[0, 255, 0], type="lower", swap="right_ankle"),
        16: dict(name="right_ankle", id=16, color=[255, 128, 0], type="lower", swap="left_ankle"),
    },
    skeleton_info={
        0: dict(link=("left_ankle", "left_knee"), id=0, color=[0, 255, 0]),
        1: dict(link=("left_knee", "left_hip"), id=1, color=[0, 255, 0]),
        2: dict(link=("right_ankle", "right_knee"), id=2, color=[255, 128, 0]),
        3: dict(link=("right_knee", "right_hip"), id=3, color=[255, 128, 0]),
        4: dict(link=("left_hip", "right_hip"), id=4, color=[51, 153, 255]),
        5: dict(link=("left_shoulder", "left_hip"), id=5, color=[51, 153, 255]),
        6: dict(link=("right_shoulder", "right_hip"), id=6, color=[51, 153, 255]),
        7: dict(link=("left_shoulder", "right_shoulder"), id=7, color=[51, 153, 255]),
        8: dict(link=("left_shoulder", "left_elbow"), id=8, color=[0, 255, 0]),
        9: dict(link=("right_shoulder", "right_elbow"), id=9, color=[255, 128, 0]),
        10: dict(link=("left_elbow", "left_wrist"), id=10, color=[0, 255, 0]),
        11: dict(link=("right_elbow", "right_wrist"), id=11, color=[255, 128, 0]),
        12: dict(link=("nose", "head_bottom"), id=12, color=[51, 153, 255]),
        13: dict(link=("nose", "head_top"), id=13, color=[51, 153, 255]),
        14: dict(link=("head_bottom", "left_shoulder"), id=14, color=[51, 153, 255]),
        15: dict(link=("head_bottom", "right_shoulder"), id=15, color=[51, 153, 255]),
    },
    joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5],
    sigmas=[
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
)