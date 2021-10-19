dataset_info = dict(
    dataset_name='st_rail_gesture',
    keypoint_info={
        0:
            dict(name='left_wrist',
                 id=0,
                 color=[51, 153, 255],
                 type='upper',
                 swap='right_wrist'),
        1:
            dict(
                name='left_elbow',
                id=1,
                color=[51, 153, 255],
                type='upper',
                swap='right_elbow'),
        2:
            dict(
                name='left_shoulder',
                id=2,
                color=[51, 153, 255],
                type='upper',
                swap='right_shoulder'),
        3:
            dict(
                name='head',
                id=3,
                color=[51, 153, 255],
                type='upper',
                swap=''),
        4:
            dict(
                name='neck',
                id=4,
                color=[51, 153, 255],
                type='upper',
                swap=''),
        5:
            dict(
                name='right_shoulder',
                id=5,
                color=[0, 255, 0],
                type='upper',
                swap='left_shoulder'),
        6:
            dict(
                name='right_elbow',
                id=6,
                color=[255, 128, 0],
                type='upper',
                swap='left_elbow'),
        7:
            dict(
                name='right_wrist',
                id=7,
                color=[0, 255, 0],
                type='upper',
                swap='left_wrist'),
        8:
            dict(
                name='butt',
                id=8,
                color=[255, 128, 0],
                type='lower',
                swap=''),
        9:
            dict(
                name='right_knee',
                id=9,
                color=[0, 255, 0],
                type='lower',
                swap='left_knee'),
        10:
            dict(
                name='right_ankle',
                id=10,
                color=[255, 128, 0],
                type='lower',
                swap='left_ankle'),
        11:
            dict(
                name='left_knee',
                id=11,
                color=[0, 255, 0],
                type='lower',
                swap='right_knee'),
        12:
            dict(
                name='left_ankle',
                id=12,
                color=[255, 128, 0],
                type='lower',
                swap='right_ankle')
    },
    skeleton_info={
        0:
            dict(link=('left_wrist', 'left_elbow'), id=0, color=[0, 255, 0]),
        1:
            dict(link=('left_elbow', 'left_shoulder'), id=1, color=[0, 255, 0]),
        2:
            dict(link=('left_shoulder', 'neck'), id=2, color=[255, 128, 0]),
        3:
            dict(link=('head', 'neck'), id=3, color=[255, 0, 0]),
        4:
            dict(link=('neck', 'right_shoulder'), id=4, color=[255, 128, 0]),
        5:
            dict(link=('right_shoulder', 'right_elbow'), id=5, color=[51, 153, 255]),
        6:
            dict(link=('right_elbow', 'right_wrist'), id=6, color=[51, 153, 255]),
        7:
            dict(link=('neck', 'butt'), id=7, color=[128, 0, 128]),
        8:
            dict(link=('butt', 'right_knee'), id=8, color=[255, 255, 0]),
        9:
            dict(link=('right_knee', 'right_ankle'), id=9, color=[255, 255, 0]),
        10:
            dict(link=('butt', 'left_knee'), id=10, color=[0, 255, 255]),
        11:
            dict(link=('left_knee', 'left_ankle'), id=11, color=[0, 255, 255])
    },
    joint_weights=[
        1.5, 1.2, 1.2, 1., 1., 1.2, 1.2, 1.5, 1., 1., 1., 1., 1.,
    ],
    sigmas=[
        0.062, 0.072, 0.079,
        0.08, 0.089,
        0.079, 0.072, 0.062,
        0.11,
        0.087, 0.089, 0.087, 0.089
    ])
