dataset_info = dict(
    dataset_name='st_rail_gesture',
    keypoint_info={
        0:
            dict(name='left_ankle',
                 id=0,
                 color=[51, 153, 255],
                 type='lower',
                 swap='right_ankle'),
        1:
            dict(
                name='left_knee',
                id=1,
                color=[51, 153, 255],
                type='lower',
                swap='right_knee'),
        2:
            dict(
                name='left_hip',
                id=2,
                color=[51, 153, 255],
                type='lower',
                swap='right_hip'),
        3:
            dict(
                name='left_wrist',
                id=3,
                color=[51, 153, 255],
                type='upper',
                swap='right_wrist'),
        4:
            dict(
                name='left_elbow',
                id=4,
                color=[51, 153, 255],
                type='upper',
                swap='right_elbow'),
        5:
            dict(
                name='left_shoulder',
                id=5,
                color=[0, 255, 0],
                type='upper',
                swap='right_shoulder'),
        6:
            dict(
                name='head',
                id=6,
                color=[255, 128, 0],
                type='upper',
                swap=''),
        7:
            dict(
                name='right_shoulder',
                id=7,
                color=[0, 255, 0],
                type='upper',
                swap='left_shoulder'),
        8:
            dict(
                name='right_elbow',
                id=8,
                color=[255, 128, 0],
                type='upper',
                swap='left_elbow'),
        9:
            dict(
                name='right_wrist',
                id=9,
                color=[0, 255, 0],
                type='upper',
                swap='left_wrist'),
        10:
            dict(
                name='right_hip',
                id=10,
                color=[255, 128, 0],
                type='lower',
                swap='left_hip'),
        11:
            dict(
                name='right_knee',
                id=11,
                color=[0, 255, 0],
                type='lower',
                swap='left_knee'),
        12:
            dict(
                name='right_ankle',
                id=12,
                color=[255, 128, 0],
                type='lower',
                swap='left_ankle')
    },
    skeleton_info={
        0:
            dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
            dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
            dict(link=('left_wrist', 'left_elbow'), id=2, color=[255, 128, 0]),
        3:
            dict(link=('left_elbow', 'left_shoulder'), id=3, color=[255, 128, 0]),
        4:
            dict(link=('left_shoulder', 'right_shoulder'), id=4, color=[51, 153, 255]),
        5:
            dict(link=('right_shoulder', 'right_elbow'), id=5, color=[51, 153, 255]),
        6:
            dict(link=('right_elbow', 'right_wrist'), id=6, color=[51, 153, 255]),
        7:
            dict(link=('right_hip', 'right_knee'), id=7, color=[51, 153, 255]),
        8:
            dict(link=('right_knee', 'right_ankle'), id=8, color=[0, 255, 0]),
        9:
            dict(link=('left_hip', 'right_hip'), id=9, color=[255, 128, 0])
    },
    joint_weights=[
        1., 1., 1., 1.5, 1.2, 1.2, 1., 1.2, 1.2, 1.5, 1., 1., 1.
    ],
    sigmas=[
        0.089, 0.087, 0.107, 0.062, 0.072, 0.079,
        0.089,
        0.079, 0.072, 0.062, 0.107, 0.087, 0.089
    ])
