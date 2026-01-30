import os
from glob import glob
from setuptools import setup

package_name = 'manipulators'


def collect_data_files(source_dir):
    """Recursively collect all files from a directory for installation."""
    file_list = []
    for root, _, files in os.walk(source_dir):
        if files:
            install_dir = os.path.join('share', package_name, root)
            file_list.append((install_dir, [os.path.join(root, f) for f in files]))
    return file_list


data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
]
data_files += collect_data_files('assets')

setup(
    name=package_name,
    version='0.1.0',
    packages=['src', 'src.detectors', 'scripts'],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'control_node = src.control_node:main',
            'keyboard_teleop = src.keyboard_teleop:main',
            'object_detection_node = src.object_detection_node:main',
            'capture_images = scripts.capture_images:main',
            'calibrate_intrinsics = scripts.calibrate_intrinsics:main',
            'calibrate_extrinsics = scripts.calibrate_extrinsics:main',
        ],
    },
)
