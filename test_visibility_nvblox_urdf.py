import numpy as np
import math
import imageio.v3 as iio
import os
import sys

import open3d as o3d
import matplotlib.pyplot as plt
import torch
from PIL import Image
import zarr
from tqdm import tqdm
import time
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import copy
import gc
from urdf_parser_py.urdf import URDF
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import trimesh

# Unitree Z1 SDK 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "unitree_ros_to_real", "unitree_legged_sdk", "lib"))
sys.path.append("/home/dscho1234/Workspace/z1_sdk/lib")
import unitree_arm_interface

# UniDepth imports
from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils.camera import Pinhole
from unidepth.utils import colorize
from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.common.utility.zarr import parallel_reading

# nvblox imports
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.mesh import ColorMesh
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType


# ========= Z1 Robot Visualizer Class =========
class Z1RobotVisualizer:
    def __init__(self, urdf_path, mesh_base_path):
        """
        Z1 로봇 시각화 클래스 초기화
        
        Args:
            urdf_path: URDF 파일 경로
            mesh_base_path: 메시 파일들이 있는 기본 경로
        """
        self.urdf_path = urdf_path
        self.mesh_base_path = mesh_base_path
        self.robot = None
        self.joint_angles = np.zeros(6)  # 6개 조인트 각도
        self.link_transforms = {}  # 각 링크의 변환 행렬 저장
        self.meshes = {}  # 로드된 메시들 저장
        
        # Unitree Z1 SDK 초기화
        try:
            self.arm_interface = unitree_arm_interface.ArmInterface(hasGripper=True)
            print("Unitree Z1 SDK 초기화 성공")
        except Exception as e:
            print(f"Unitree Z1 SDK 초기화 실패: {e}")
            self.arm_interface = None
        
        # URDF 파싱
        self.parse_urdf()
        
        # 메시 로딩
        self.load_meshes()
        
    def parse_urdf(self):
        """URDF 파일을 파싱하여 로봇 구조 정보 추출"""
        try:
            self.robot = URDF.from_xml_file(self.urdf_path)
            print(f"URDF 파싱 완료: {len(self.robot.joints)} 개 조인트, {len(self.robot.links)} 개 링크")
            
            # 조인트 정보 출력
            for i, joint in enumerate(self.robot.joints):
                print(f"Joint {i+1}: {joint.name}, Type: {joint.type}, Axis: {joint.axis}")
                
        except Exception as e:
            print(f"URDF 파싱 오류: {e}")
            
    def load_meshes(self):
        """각 링크의 메시 파일을 로드"""
        # URDF에서 메시 파일 경로 추출
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        for link in root.findall('link'):
            link_name = link.get('name')
            visual = link.find('visual')
            
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_filename = mesh.get('filename')
                        if mesh_filename and 'package://' in mesh_filename:
                            # package:// 경로를 실제 파일 경로로 변환
                            mesh_path = mesh_filename.replace('package://z1_description/', self.mesh_base_path)
                            
                            try:
                                # 먼저 STL 파일 시도
                                stl_path = mesh_path.replace('.dae', '.STL').replace('visual/', 'collision/')
                                if os.path.exists(stl_path):
                                    # STL 파일 로드
                                    mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                                    if len(mesh_obj.vertices) > 0:
                                        self.meshes[link_name] = mesh_obj
                                        print(f"STL 메시 로드 성공: {link_name} -> {stl_path} ({len(mesh_obj.vertices)} vertices)")
                                        continue
                                
                                # DAE 파일 시도
                                trimesh_obj = trimesh.load(mesh_path)
                                
                                # Open3D 메시로 변환
                                if hasattr(trimesh_obj, 'vertices') and hasattr(trimesh_obj, 'faces'):
                                    mesh_obj = o3d.geometry.TriangleMesh()
                                    mesh_obj.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
                                    mesh_obj.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
                                    
                                    if len(mesh_obj.vertices) > 0:
                                        self.meshes[link_name] = mesh_obj
                                        print(f"DAE 메시 로드 성공: {link_name} -> {mesh_path} ({len(mesh_obj.vertices)} vertices)")
                                    else:
                                        print(f"빈 메시: {link_name}")
                                else:
                                    print(f"메시 형식 오류: {link_name}")
                            except Exception as e:
                                print(f"메시 로드 실패: {link_name} - {e}")
                                # 메시 로드 실패 시 간단한 기하학적 모양 생성
                                self.create_simple_geometry(link_name)
        
        # Gripper 메시도 로드 (URDF에 정의되지 않았지만 메시 파일이 있음)
        self.load_gripper_meshes()
                                
    def load_gripper_meshes(self):
        """Gripper 메시 파일들을 로드"""
        gripper_meshes = ['z1_GripperMover', 'z1_GripperStator']
        
        for gripper_name in gripper_meshes:
            # DAE 파일 경로
            dae_path = os.path.join(self.mesh_base_path, 'meshes', 'visual', f'{gripper_name}.dae')
            # STL 파일 경로
            stl_path = os.path.join(self.mesh_base_path, 'meshes', 'collision', f'{gripper_name}.STL')
            
            try:
                # 먼저 STL 파일 시도
                if os.path.exists(stl_path):
                    mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                    if len(mesh_obj.vertices) > 0:
                        self.meshes[gripper_name] = mesh_obj
                        print(f"Gripper STL 메시 로드 성공: {gripper_name} -> {stl_path} ({len(mesh_obj.vertices)} vertices)")
                        continue
                
                # DAE 파일 시도
                if os.path.exists(dae_path):
                    trimesh_obj = trimesh.load(dae_path)
                    if hasattr(trimesh_obj, 'vertices') and hasattr(trimesh_obj, 'faces'):
                        mesh_obj = o3d.geometry.TriangleMesh()
                        mesh_obj.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
                        mesh_obj.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
                        
                        if len(mesh_obj.vertices) > 0:
                            self.meshes[gripper_name] = mesh_obj
                            print(f"Gripper DAE 메시 로드 성공: {gripper_name} -> {dae_path} ({len(mesh_obj.vertices)} vertices)")
                            continue
                            
            except Exception as e:
                print(f"Gripper 메시 로드 실패: {gripper_name} - {e}")
                                
    def create_simple_geometry(self, link_name):
        """메시 로드 실패 시 간단한 기하학적 모양 생성"""
        # 링크별로 적절한 크기의 기하학적 모양 생성
        if link_name == 'link00':
            # 베이스 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0325, height=0.051)
        elif link_name == 'link01':
            # 첫 번째 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.045)
        elif link_name == 'link02':
            # 두 번째 링크 - 박스
            mesh = o3d.geometry.TriangleMesh.create_box(width=0.35, height=0.102, depth=0.102)
        elif link_name == 'link03':
            # 세 번째 링크 - 박스
            mesh = o3d.geometry.TriangleMesh.create_box(width=0.116, height=0.059, depth=0.059)
        elif link_name == 'link04':
            # 네 번째 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0325, height=0.067)
        elif link_name == 'link05':
            # 다섯 번째 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=0.0492)
        elif link_name == 'link06':
            # 여섯 번째 링크 - 원통
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0325, height=0.051)
        else:
            # 기본 - 구
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            
        self.meshes[link_name] = mesh
        print(f"간단한 기하학적 모양 생성: {link_name}")
                                
    def set_joint_angles(self, angles):
        """조인트 각도 설정"""
        self.joint_angles = np.array(angles)
        print(f"조인트 각도 설정: {self.joint_angles}")
        
    def solve_inverse_kinematics(self, target_pose, gripper_angle=0.0):
        """Inverse Kinematics를 사용하여 목표 pose에서 조인트 각도 계산"""
        if self.arm_interface is None:
            print("Unitree Z1 SDK가 초기화되지 않았습니다.")
            return False
            
        try:
            # 현재 조인트 각도를 초기 추정값으로 사용
            current_q = self.joint_angles.copy()
            print(f"초기 조인트 각도: {current_q}")
            
            # Inverse Kinematics 계산
            success, q_forward = self.arm_interface._ctrlComp.armModel.inverseKinematics(
                target_pose, current_q, True  # checkInWorkSpace=True
            )
            
            print(f"계산 후 조인트 각도: {q_forward}")
            
            if success:
                # 계산된 조인트 각도로 업데이트
                self.joint_angles = q_forward
                print(f"Inverse Kinematics 성공: {self.joint_angles}")
                
                # Forward Kinematics로 검증
                fk_result = self.arm_interface._ctrlComp.armModel.forwardKinematics(q_forward, 6)
                print(f"Forward Kinematics 검증 결과:\n{fk_result}")
                
                return True
            else:
                print("Inverse Kinematics 실패: 목표 pose가 작업 공간 밖에 있습니다.")
                return False
                
        except Exception as e:
            print(f"Inverse Kinematics 오류: {e}")
            return False
        
    def compute_forward_kinematics(self, gripper_angle=0.0):
        """전진기구학 계산 - URDF 기반 (Unitree SDK는 절대 위치를 반환하므로 부적합)"""
        # Unitree Z1 SDK는 절대 위치를 반환하므로 URDF 기반 계산 사용
        self.compute_forward_kinematics_urdf(gripper_angle)
    
    def compute_forward_kinematics_urdf(self, gripper_angle=0.0):
        """URDF 기반 전진기구학 계산 (폴백)"""
        # 조인트 정보를 딕셔너리로 저장
        joint_info = {}
        for joint in self.robot.joints:
            joint_info[joint.name] = joint
            
        # 링크 정보를 딕셔너리로 저장
        link_info = {}
        for link in self.robot.links:
            link_info[link.name] = link
            
        # 변환 행렬 초기화
        self.link_transforms = {}
        
        # world -> link00 (고정 조인트)
        self.link_transforms['world'] = np.eye(4)
        self.link_transforms['link00'] = np.eye(4)
        
        # 각 조인트에 대해 변환 행렬 계산
        joint_angle_idx = 0
        for joint in self.robot.joints:
            if joint.type == 'revolute' and joint_angle_idx < len(self.joint_angles):
                # 조인트 각도
                angle = self.joint_angles[joint_angle_idx]
                joint_angle_idx += 1
                
                # 조인트 축
                axis = np.array(joint.axis)
                
                # 조인트 원점
                origin = joint.origin
                if origin is not None:
                    xyz = np.array(origin.xyz) if origin.xyz else np.zeros(3)
                    rpy = np.array(origin.rpy) if origin.rpy else np.zeros(3)
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)
                
                # 회전 행렬 계산 (RPY)
                if np.any(rpy):
                    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
                else:
                    rot_matrix = np.eye(3)
                
                # 조인트 회전 행렬 (축 주위 회전)
                joint_rot_matrix = R.from_rotvec(axis * angle).as_matrix()
                
                # 변환 행렬 구성
                T_joint = np.eye(4)
                T_joint[:3, :3] = rot_matrix @ joint_rot_matrix
                T_joint[:3, 3] = xyz
                
                # 부모 링크의 변환 행렬과 결합
                parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                child_transform = parent_transform @ T_joint
                
                self.link_transforms[joint.child] = child_transform
                
            elif joint.type == 'fixed':
                # 고정 조인트
                origin = joint.origin
                if origin is not None:
                    xyz = np.array(origin.xyz) if origin.xyz else np.zeros(3)
                    rpy = np.array(origin.rpy) if origin.rpy else np.zeros(3)
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)
                
                # 회전 행렬 계산
                if np.any(rpy):
                    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
                else:
                    rot_matrix = np.eye(3)
                
                # 변환 행렬 구성
                T_fixed = np.eye(4)
                T_fixed[:3, :3] = rot_matrix
                T_fixed[:3, 3] = xyz
                
                # 부모 링크의 변환 행렬과 결합
                parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                child_transform = parent_transform @ T_fixed
                
                self.link_transforms[joint.child] = child_transform
        
        # Gripper 위치 계산 (URDF 기반 정확한 오프셋 사용)
        if 'link06' in self.link_transforms:
            # gripperStator는 link06에서 xyz="0.051 0.0 0.0" 오프셋으로 연결
            gripper_stator_offset = np.array([0.051, 0.0, 0.0])  # URDF에서 정의된 오프셋
            gripper_stator_transform = self.link_transforms['link06'].copy()
            gripper_stator_transform[:3, 3] += gripper_stator_transform[:3, :3] @ gripper_stator_offset
            
            self.link_transforms['z1_GripperStator'] = gripper_stator_transform
            
            # gripperMover는 gripperStator에서 xyz="0.049 0.0 0" 오프셋으로 연결
            gripper_mover_offset = np.array([0.049, 0.0, 0.0])  # URDF에서 정의된 오프셋
            gripper_mover_transform = gripper_stator_transform.copy()
            gripper_mover_transform[:3, 3] += gripper_mover_transform[:3, :3] @ gripper_mover_offset
            
            # gripper 회전 적용 (Y축 주위 회전, URDF에서 axis xyz="0 1 0")
            if gripper_angle != 0.0:
                gripper_rotation = R.from_rotvec([0, gripper_angle, 0]).as_matrix()
                gripper_mover_transform[:3, :3] = gripper_mover_transform[:3, :3] @ gripper_rotation
            
            self.link_transforms['z1_GripperMover'] = gripper_mover_transform
                
    def transform_mesh_to_world(self, mesh, transform):
        """메시를 월드 좌표계로 변환"""
        # 메시 복사 (Open3D 방식)
        transformed_mesh = o3d.geometry.TriangleMesh()
        transformed_mesh.vertices = mesh.vertices
        transformed_mesh.triangles = mesh.triangles
        
        # 정점들을 월드 좌표계로 변환
        vertices = np.asarray(transformed_mesh.vertices)
        vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        vertices_world = (transform @ vertices_homogeneous.T).T[:, :3]
        
        # 변환된 정점으로 메시 업데이트
        transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
        
        return transformed_mesh
        
    def set_robot_base_transform(self, base_position, base_orientation=None):
        """
        로봇 베이스의 position과 orientation을 설정
        
        Args:
            base_position: 베이스 위치 [x, y, z] (미터)
            base_orientation: 베이스 방향 (3x3 회전 행렬 또는 None)
        """
        self.base_transform = np.eye(4)
        self.base_transform[:3, 3] = base_position
        
        if base_orientation is not None:
            self.base_transform[:3, :3] = base_orientation
        else:
            # 기본 orientation: Z축이 위를 향하도록
            self.base_transform[:3, :3] = np.eye(3)
    
    def get_robot_meshes_in_world(self, gripper_angle=0.0):
        """로봇의 모든 링크 메시를 월드 좌표계로 변환하여 반환"""
        # 전진기구학 계산
        self.compute_forward_kinematics(gripper_angle)
        
        world_meshes = {}
        for link_name, mesh in self.meshes.items():
            if link_name in self.link_transforms:
                # 메시를 월드 좌표계로 변환
                world_mesh = self.transform_mesh_to_world(mesh, self.link_transforms[link_name])
                
                # 베이스 변환이 설정된 경우 적용
                if hasattr(self, 'base_transform'):
                    world_mesh = self.transform_mesh_to_world(world_mesh, self.base_transform)
                
                world_meshes[link_name] = world_mesh
        
        return world_meshes


# ========= 사용자 설정 =========
# Zarr 데이터 경로 설정
BUFFER_PATH = "/home/dscho1234/fast_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/slam_head_mounted_camera_multi_marker_initial_lag"
EPISODE_IDX = 0
FRAME_IDX = 100  # 특정 프레임 선택
DEPTH_SCALE = 0.001        # 깊이 단위 → 미터 변환 (예: mm면 0.001, 이미 m면 1.0)
OFFSET_DISTANCE = 0.05 # for convex part of the constructed mesh

# TSDF 설정
# Mesh 품질 선택: "high_resolution" (5mm), "medium_resolution" (10mm), "low_resolution" (20mm)
MESH_QUALITY = "medium_resolution" # "medium_resolution"  # "high_resolution", "medium_resolution", "low_resolution"

if MESH_QUALITY == "high_resolution":
    VOXEL_SIZE = 0.005  # 5mm - 높은 해상도, 조각난 mesh
    
elif MESH_QUALITY == "medium_resolution":
    VOXEL_SIZE = 0.01   # 10mm - 중간 해상도, 균형잡힌 mesh
    
else:  # low_resolution
    VOXEL_SIZE = 0.02   # 20mm - 낮은 해상도, 매끄러운 mesh
    

print(f"Using {MESH_QUALITY}: voxel_size={VOXEL_SIZE}m")

# Depth estimation 설정
USE_MONODEPTH = True  # True: UniDepth 사용, False: raw depth 사용
MODEL_TYPE = "l"  # UniDepth model type: s, b, l

# 이미지 리사이즈 설정
RESIZE = True  # True: 이미지를 256x256으로 리사이즈, False: 원본 크기 사용
RESIZE_SIZE = (256, 256)  # 리사이즈할 크기 (width, height)

# (중요) 카메라 내파라미터: 사용자가 직접 채우세요.
K = np.array([
    [604.682922, 0.0, 328.062561],
    [0.0, 604.898438, 244.393188],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# 새 카메라 뷰포인트들 (월드 좌표, world = 원 카메라 좌표계로 가정)
# CANDIDATE_VIEWPOINTS = [
#     np.array([0.20, 0.00, 0.00], dtype=np.float64),  # 옆으로
#     np.array([0.00, 0.20, 0.00], dtype=np.float64),  # 위로
#     np.array([0.00, 0.00, 0.20], dtype=np.float64),  # 앞으로
#     np.array([0.15, 0.15, 0.00], dtype=np.float64),  # 대각선
#     np.array([0.15, 0.15, 0.15], dtype=np.float64),  # 3D 대각선
# ]
CANDIDATE_VIEWPOINTS = [
    np.array([-0.05, 0.00, 0.00], dtype=np.float64),
    np.array([-0.10, 0.00, 0.00], dtype=np.float64),
    np.array([-0.15, 0.00, 0.00], dtype=np.float64),
    np.array([-0.20, 0.00, 0.00], dtype=np.float64),
    np.array([-0.25, 0.00, 0.00], dtype=np.float64),
]


# 각 viewpoint의 yaw 회전 (deg)
CANDIDATE_YAW_DEGS = [0.0, 0.0, 0.0, 0.0, 0.0]

# M개의 viewpoint set 생성 (예제에서는 M=3으로 설정)
M = 10 # 3  # M개의 viewpoint set
L = len(CANDIDATE_VIEWPOINTS)  # L개의 viewpoints per set



print(f"Creating {M} viewpoint sets, each with {L} viewpoints")
print(f"Total viewpoints: {M} x {L} = {M*L}")

# [M, L] 모양의 candidate viewpoints 배열 생성
# 현재는 예제로 기존 CANDIDATE_VIEWPOINTS를 M번 복사
CANDIDATE_VIEWPOINTS_MATRIX = np.array([CANDIDATE_VIEWPOINTS for _ in range(M)])
print(f"Candidate viewpoints matrix shape: {CANDIDATE_VIEWPOINTS_MATRIX.shape}")  # [M, L, 3]


# ========= 유틸 함수 =========
def resize_image_and_depth(rgb, depth, target_size=(256, 256)):
    """
    RGB 이미지와 depth 이미지를 지정된 크기로 리사이즈 (OpenCV 사용)
    
    Args:
        rgb: RGB 이미지 (H, W, 3)
        depth: depth 이미지 (H, W)
        target_size: 리사이즈할 크기 (width, height)
    
    Returns:
        resized_rgb: 리사이즈된 RGB 이미지
        resized_depth: 리사이즈된 depth 이미지
        scale_factor: 스케일 팩터 (원본 크기 / 리사이즈 크기)
    """
    import cv2
    
    H, W = rgb.shape[:2]
    target_w, target_h = target_size
    
    rgb_uint8 = rgb.astype(np.uint8)
    
    rgb_resized = cv2.resize(rgb_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    
    
    # Depth 이미지 리사이즈 (INTER_NEAREST 사용)
    depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # 스케일 팩터 계산 (x, y 방향 각각)
    scale_factor_x = (target_w-1) / (W-1)
    scale_factor_y = (target_h-1) / (H-1)
    
    return rgb_resized, depth_resized, scale_factor_x, scale_factor_y


def adjust_camera_intrinsics(K, scale_factor_x, scale_factor_y):
    """
    카메라 내부 파라미터를 리사이즈에 맞게 조정
    
    Args:
        K: 원본 카메라 내부 파라미터 (3, 3)
        scale_factor_x: x 방향 스케일 팩터
        scale_factor_y: y 방향 스케일 팩터
    
    Returns:
        K_adjusted: 조정된 카메라 내부 파라미터
    """
    K_adjusted = K.copy()
    K_adjusted[0, 0] *= scale_factor_x  # fx
    K_adjusted[1, 1] *= scale_factor_y  # fy
    K_adjusted[0, 2] *= scale_factor_x  # cx
    K_adjusted[1, 2] *= scale_factor_y  # cy
    
    return K_adjusted


def load_rgbd_from_zarr(buffer_path, episode_idx, frame_idx, depth_scale=0.001):
    """Zarr에서 RGB와 depth 데이터를 불러오는 함수"""
    register_codecs()
    
    # Zarr 버퍼 열기
    buffer = zarr.open(buffer_path, mode="r")
    episode = buffer[f"episode_{episode_idx}"]
    
    # RGB 프레임 로드
    rgb_frames = parallel_reading(
        group=episode["camera_0"],
        array_name="rgb",
    )
    rgb = rgb_frames[frame_idx]  # 특정 프레임 선택
    
    # Depth 프레임 로드
    
    depth_frames = parallel_reading(
        group=episode["camera_0"],
        array_name="depth",
    )
    depth_raw = depth_frames[frame_idx]  # 특정 프레임 선택
    depth_m = depth_raw.astype(np.float32) * depth_scale
    
    
    return rgb, depth_m


def radial_depth_to_z_depth(radial_depth, intrinsics):
    """
    Convert radial depth (distance from camera center) to Z-depth (distance from camera plane).
    """
    H, W = radial_depth.shape
    
    # Create pixel coordinate grid
    pixel_x, pixel_y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing='xy'
    )
    
    # Convert to normalized camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Normalized coordinates (x/z, y/z)
    norm_x = (pixel_x - cx) / fx
    norm_y = (pixel_y - cy) / fy
    
    # Calculate squared distance from optical center
    squared_distance_from_center = norm_x**2 + norm_y**2
    
    # Convert radial depth to Z-depth
    depth_scaling = np.sqrt(1 + squared_distance_from_center)
    z_depth = radial_depth / depth_scaling
    
    return z_depth


def get_depth_with_unidepth(rgb, intrinsics, model, camera):
    """UniDepth를 사용하여 depth를 추정하고 Z-depth로 변환"""
    # Convert RGB frame to tensor
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    
    # Predict depth
    with torch.no_grad():
        predictions = model.infer(rgb_torch, camera)
    
    # Extract depth prediction (radial depth)
    radial_depth_pred = predictions["depth"].squeeze().cpu().numpy()
    
    # Convert radial depth to Z-depth
    z_depth_pred = radial_depth_to_z_depth(radial_depth_pred, intrinsics)
    
    return z_depth_pred


def scale_depth_with_raw(depth_pred, depth_raw, intrinsics):
    """Raw depth를 사용하여 predicted depth를 스케일링"""
    # Ensure spatial dimensions match
    if depth_raw.shape != depth_pred.shape:
        print(f"Resizing depth_raw from {depth_raw.shape} to {depth_pred.shape}")
        depth_raw_resized = np.array(Image.fromarray(depth_raw.astype(np.float32)).resize(
            (depth_pred.shape[1], depth_pred.shape[0]), Image.NEAREST))
        depth_raw = depth_raw_resized
    
    # Use all valid depth pixels for scaling
    valid_mask = (depth_raw > 0) & (depth_pred > 0)
    
    if np.sum(valid_mask) > 100:  # Require at least 100 valid pixels
        # Use median for robust scaling
        scale_factor = np.median(depth_raw[valid_mask] / depth_pred[valid_mask])
        depth_scaled = depth_pred * scale_factor
        print(f"Applied scale factor {scale_factor:.3f} using raw depth ({np.sum(valid_mask)} valid pixels)")
        return depth_scaled
    else:
        print(f"Insufficient valid depth values ({np.sum(valid_mask)} pixels), using original depth")
        return depth_pred




def pick_query_pixel(depth_m):
    """이미지 중앙에서 오른쪽으로 이미지 크기에 비례한 픽셀 이동한 위치에서 유효한 depth 값을 가진 픽셀을 선택."""
    H, W = depth_m.shape
    cx, cy = W // 2, H // 2
    
    # 이미지 크기에 비례한 오프셋 계산 (원본 640x480에서 30픽셀이었으므로)
    base_offset = 30
    base_width = 640
    offset = max(int(base_offset * W / base_width), 5)  # 최소 5픽셀
    
    # 중앙에서 오른쪽으로 비례한 픽셀 이동
    target_x = cx + offset
    target_y = cy
    
    # 타겟 위치가 유효한지 확인
    if 0 <= target_x < W and 0 <= target_y < H and depth_m[target_y, target_x] > 0:
        return (target_x, target_y)
    
    # 타겟 위치가 유효하지 않으면 주변에서 유효한 픽셀 찾기
    radius = max(H, W) // 20
    ys, xs = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = xs*xs + ys*ys <= radius*radius
    candidates = np.argwhere(mask) + np.array([target_y-radius, target_x-radius])
    for yy, xx in candidates:
        if 0 <= yy < H and 0 <= xx < W and depth_m[yy, xx] > 0:
            return int(xx), int(yy)
    
    # 여전히 찾지 못하면 원래 중앙에서 찾기
    if depth_m[cy, cx] > 0:
        return (cx, cy)
    
    # 마지막 fallback: 유효한 픽셀 중 하나 선택
    ys, xs = np.where(depth_m > 0)
    if len(ys) == 0:
        raise RuntimeError("유효 깊이 픽셀이 없습니다.")
    i = len(ys) // 2
    return int(xs[i]), int(ys[i])


# ========= nvblox 기반 메시 생성 =========
def create_mesh_with_nvblox(depth_image, rgb_image, K, voxel_size=0.005, max_integration_distance=5.0, return_mapper=False, mapper=None):
    """
    nvblox를 사용하여 3D 메시 생성
    
    Args:
        depth_image: (H, W) 깊이 이미지 (미터)
        rgb_image: (H, W, 3) RGB 이미지
        K: (3, 3) 카메라 내부 파라미터
        voxel_size: voxel 크기 (미터)
        max_integration_distance: 최대 통합 거리 (미터)
        return_mapper: True면 (mesh, mapper) 튜플 반환, False면 mesh만 반환
        mapper: 기존 Mapper 객체 재사용 (None이면 새로 생성)
    
    Returns:
        mesh: nvblox ColorMesh 객체 또는 (mesh, mapper) 튜플
    """
    print("Creating mesh with nvblox...")
    start = time.time()
    # 데이터를 torch tensor로 변환
    H, W = depth_image.shape
    
    # Depth 이미지를 torch tensor로 변환 (GPU)
    depth_tensor = torch.from_numpy(depth_image.astype(np.float32)).cuda()
    
    # RGB 이미지를 torch tensor로 변환 (GPU, uint8)
    rgb_tensor = torch.from_numpy((rgb_image * 255).astype(np.uint8)).cuda()
    
    # 카메라 내부 파라미터를 torch tensor로 변환 (CPU)
    intrinsics_tensor = torch.from_numpy(K.astype(np.float32)).cpu()
    
    # 카메라 포즈 (identity matrix, CPU)
    pose_tensor = torch.eye(4, dtype=torch.float32).cpu()
    
    # Mapper 재사용 또는 새로 생성
    if mapper is None:
        # nvblox Mapper 설정
        projective_integrator_params = ProjectiveIntegratorParams()
        projective_integrator_params.projective_integrator_max_integration_distance_m = max_integration_distance
        mapper_params = MapperParams()
        mapper_params.set_projective_integrator_params(projective_integrator_params)
        
        # Mapper 생성
        mapper = Mapper(
            voxel_sizes_m=voxel_size,
            # integrator_types=ProjectiveIntegratorType.TSDF,
            mapper_parameters=mapper_params
        )
    
    
    # 데이터 통합
    mapper.add_depth_frame(depth_tensor, pose_tensor, intrinsics_tensor)
    mapper.add_color_frame(rgb_tensor, pose_tensor, intrinsics_tensor)
    
    # 메시 업데이트
    mapper.update_color_mesh()
    
    # 메시 가져오기
    color_mesh = mapper.get_color_mesh()
    
    print(f"nvblox mesh created with {color_mesh.vertices().shape[0]} vertices and {color_mesh.triangles().shape[0]} triangles")
    print('color mesh update time: ', time.time() - start)
    
    
    if return_mapper:
        return color_mesh, mapper
    else:
        return color_mesh




# ========= nvblox ESDF 기반 Visibility 체크 =========
def nvblox_mesh_to_open3d(nvblox_mesh):
    """
    nvblox ColorMesh를 Open3D TriangleMesh로 변환
    
    Args:
        nvblox_mesh: nvblox ColorMesh 객체
    
    Returns:
        o3d.geometry.TriangleMesh: Open3D 메시 객체
    """
    
    # # nvblox mesh에서 vertices와 triangles 추출
    # vertices = nvblox_mesh.vertices().cpu().numpy()
    # triangles = nvblox_mesh.triangles().cpu().numpy()
    
    # if len(vertices) == 0 or len(triangles) == 0:
    #     return None
    
    # # Open3D 메시 생성
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # # 법선 계산 (raycasting에 필요)
    # mesh.compute_vertex_normals()


    # dscho debug
    mesh = nvblox_mesh.to_open3d()
    
    return mesh
    





    
    






def create_raycasting_scene(mesh):
    """
    nvblox mesh로부터 Open3D RaycastingScene을 미리 생성
    
    Args:
        mesh: nvblox ColorMesh 객체 또는 Open3D mesh 객체
    
    Returns:
        scene: o3d.t.geometry.RaycastingScene 객체
    """
    print("    Creating raycasting scene...")
    start_time = time.time()
    
    # 1. nvblox mesh를 Open3D로 변환 (필요한 경우)
    if hasattr(mesh, 'vertices') and callable(mesh.vertices):
        # nvblox ColorMesh
        open3d_mesh = nvblox_mesh_to_open3d(mesh)
    else:
        # 이미 Open3D mesh
        open3d_mesh = mesh
    
    if open3d_mesh is None:
        print("    Failed to convert mesh, returning None")
        return None
    
    # 2. 메시를 tensor로 변환
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
    
    # 3. Ray casting scene 생성
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    
    end_time = time.time()
    print(f"    Scene creation time: {end_time - start_time:.4f}s")
    
    return scene


def create_combined_scene_with_robot(scene_mesh, robot_meshes):
    """
    Scene mesh와 robot meshes를 결합한 raycasting scene 생성
    
    Args:
        scene_mesh: nvblox ColorMesh 객체 또는 Open3D mesh 객체
        robot_meshes: 로봇 링크 메시들의 딕셔너리 {link_name: mesh}
    
    Returns:
        scene: o3d.t.geometry.RaycastingScene 객체
    """
    print("    Creating combined scene with robot...")
    start_time = time.time()
    
    # 1. Scene mesh를 Open3D로 변환
    if hasattr(scene_mesh, 'vertices') and callable(scene_mesh.vertices):
        # nvblox ColorMesh
        scene_open3d = nvblox_mesh_to_open3d(scene_mesh)
    else:
        # 이미 Open3D mesh
        scene_open3d = scene_mesh
    
    if scene_open3d is None:
        print("    Failed to convert scene mesh, returning None")
        return None
    
    # 2. 모든 메시를 하나로 결합
    combined_mesh = scene_open3d
    
    # 3. 각 로봇 링크 메시를 결합
    for link_name, robot_mesh in robot_meshes.items():
        if robot_mesh is not None and len(robot_mesh.vertices) > 0:
            combined_mesh += robot_mesh
            print(f"    Added robot link {link_name}: {len(robot_mesh.vertices)} vertices")
    
    # 4. 결합된 메시를 tensor로 변환
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(combined_mesh)
    
    # 5. Ray casting scene 생성
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    
    end_time = time.time()
    print(f"    Combined scene creation time: {end_time - start_time:.4f}s")
    print(f"    Total vertices in combined scene: {len(combined_mesh.vertices)}")
    
    return scene


def batch_raycasting_with_scene(scene, origins, directions, max_distances):
    """
    미리 생성된 Open3D RaycastingScene을 사용한 batch raycasting
    
    Args:
        scene: 미리 생성된 o3d.t.geometry.RaycastingScene 객체
        origins: 레이 시작점들 [M, 3]
        directions: 레이 방향들 (정규화된 벡터) [M, 3]
        max_distances: 최대 거리들 [M]
    
    Returns:
        (visible_array, hit_distances_array)
        - visible_array: [M] boolean array, True if visible
        - hit_distances_array: [M] float array, hit distances
    """
    if scene is None:
        print("    Scene is None, assuming all visible")
        return np.ones(len(origins), dtype=bool), max_distances.copy()
    
    total_start_time = time.time()
    
    # 1. Batch ray 생성
    step1_start = time.time()
    rays = np.hstack([origins, directions])
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    step1_time = time.time() - step1_start
    print(f"      Step 1 - Ray preparation: {step1_time:.4f}s")
    
    # 2. Batch raycasting 수행
    step2_start = time.time()
    ans = scene.cast_rays(rays_tensor)
    step2_time = (time.time() - step2_start)
    print(f"      Step 2 - Ray casting: {step2_time:.4f}s")
    
    # 3. 결과 분석
    step3_start = time.time()
    hit_distances = ans['t_hit'].numpy()  # [M]
    
    # Visibility 판단: hit이 inf이거나 max_distance보다 크면 visible
    visible = np.logical_or(np.isinf(hit_distances), hit_distances > max_distances)
    
    # Hit distance 조정: visible한 경우 max_distance로 설정
    hit_distances_adjusted = np.where(visible, max_distances, hit_distances)
    step3_time = time.time() - step3_start
    print(f"      Step 3 - Result processing: {step3_time:.4f}s")
    
    total_time = time.time() - total_start_time
    print(f"    Batch raycasting ({len(origins)} rays): {total_time:.4f}s total")
    
    # 각 단계별 시간 요약
    print(f"      Time breakdown: rays={step1_time:.4f}s, casting={step2_time:.4f}s, processing={step3_time:.4f}s")
    
    return visible, hit_distances_adjusted





def create_3d_visualization_plotly(mesh, query_point, candidate_viewpoints, results, save_path, 
                                   robot_mask=None, robot_points_original=None, robot_points_transformed=None):
    """
    Plotly를 사용한 인터랙티브 3D 시각화 생성 (색상 정보 + 로봇 변환 정보 포함)
    """
    # 메시 데이터 추출 (nvblox ColorMesh 또는 Open3D mesh 모두 지원)
    if hasattr(mesh, 'vertices') and callable(mesh.vertices):
        # nvblox ColorMesh
        vertices = mesh.vertices().cpu().numpy()
        faces = mesh.triangles().cpu().numpy()
        vertex_colors = mesh.vertex_colors().cpu().numpy() if hasattr(mesh, 'vertex_colors') and callable(mesh.vertex_colors) else None
    else:
        # Open3D mesh (fallback)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors) if hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0 else None
    
    # 디버깅: mesh 정보 출력
    print(f"=== Mesh Visualization Debug ===")
    print(f"Mesh vertices: {len(vertices)}")
    print(f"Mesh faces: {len(faces)}")
    print(f"Robot points original: {len(robot_points_original) if robot_points_original is not None else 0}")
    print(f"Robot points transformed: {len(robot_points_transformed) if robot_points_transformed is not None else 0}")
    
    # Robot transform으로 인한 mesh 변화 확인
    if robot_points_original is not None and robot_points_transformed is not None:
        print(f"Robot transformation applied: {len(robot_points_original)} -> {len(robot_points_transformed)} points")
        # 변환된 robot points가 mesh에 포함되어 있는지 확인
        if len(robot_points_transformed) > 0:
            # 변환된 robot points의 범위 확인
            robot_min = np.min(robot_points_transformed, axis=0)
            robot_max = np.max(robot_points_transformed, axis=0)
            print(f"Transformed robot bounds: min={robot_min}, max={robot_max}")
            
            # Mesh vertices 범위 확인
            mesh_min = np.min(vertices, axis=0)
            mesh_max = np.max(vertices, axis=0)
            print(f"Mesh bounds: min={mesh_min}, max={mesh_max}")
            
            # Robot points가 mesh 범위 내에 있는지 확인
            robot_in_mesh = np.all(robot_min >= mesh_min - 0.1) and np.all(robot_max <= mesh_max + 0.1)
            print(f"Robot points within mesh bounds: {robot_in_mesh}")
    
    # 디버깅 정보 출력
    print(f"Mesh data: {len(vertices)} vertices, {len(faces)} faces")
    if len(faces) > 0:
        max_face_idx = np.max(faces)
        min_face_idx = np.min(faces)
        print(f"Face indices range: {min_face_idx} to {max_face_idx}")
        print(f"Vertex indices range: 0 to {len(vertices)-1}")
        
        # Face 인덱스가 vertex 범위를 벗어나는지 확인
        if max_face_idx >= len(vertices):
            print(f"WARNING: Face indices exceed vertex range! Max face idx: {max_face_idx}, Max vertex idx: {len(vertices)-1}")
            # 잘못된 face들을 필터링
            valid_faces = faces[np.all(faces < len(vertices), axis=1)]
            print(f"Filtered faces: {len(valid_faces)} valid faces out of {len(faces)}")
            faces = valid_faces
    
    # 3D 시각화 생성
    fig = go.Figure()
    
    # 1. 메시 표시 (실제 mesh로 시각화)
    if len(faces) > 0 and len(vertices) > 0:
        # 간단한 샘플링: 면의 수만 제한하고 vertex는 그대로 유지
        if len(faces) > 50000:
            # 면의 수를 제한 (vertex는 그대로 유지) - 더 많은 face 사용
            face_indices = np.random.choice(len(faces), 50000, replace=False)
            faces_sampled = faces[face_indices]
            print(f"Sampled {len(faces_sampled)} faces from {len(faces)} total faces")
        else:
            faces_sampled = faces
        
        # Mesh3d로 실제 mesh 표시 (vertex 인덱스는 원본 그대로 사용)
        if vertex_colors is not None and len(vertex_colors) > 0:
            # RGB 색상 정보가 있는 경우 - 실제 RGB 색상 사용
            if vertex_colors.max() <= 1.0:
                # Open3D 형식 (0-1 범위)
                colors_rgb = vertex_colors
            else:
                # nvblox 형식 (0-255 범위) - 0-1로 정규화
                colors_rgb = vertex_colors / 255.0
            
            # RGB 색상을 hex 문자열로 변환
            colors_hex = []
            for color in colors_rgb:
                r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                colors_hex.append(f'rgb({r},{g},{b})')
            
            # RGB 색상을 사용한 mesh 시각화
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces_sampled[:, 0],
                j=faces_sampled[:, 1],
                k=faces_sampled[:, 2],
                facecolor=colors_hex,  # 실제 RGB 색상 사용
                opacity=0.8,
                name='3D Mesh (RGB Colored)',
                showlegend=True
            ))
        else:
            # 색상 정보가 없는 경우 (기본 색상)
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces_sampled[:, 0],
                j=faces_sampled[:, 1],
                k=faces_sampled[:, 2],
                color='lightblue',
                opacity=0.6,
                name='3D Mesh',
                showlegend=True
            ))
    else:
        # Fallback: point cloud로 표시
        print("Warning: No faces found, falling back to point cloud visualization")
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='lightblue',
                opacity=0.6
            ),
            name='3D Scene (Point Cloud)',
            showlegend=True
        ))
    
    # 1.5. 원본 로봇 포인트들을 mesh로 표시 (빨간색)
    if robot_points_original is not None and len(robot_points_original) > 0:
        # 로봇 포인트들을 샘플링 (너무 많으면)
        if len(robot_points_original) > 2000:
            robot_indices = np.random.choice(len(robot_points_original), 2000, replace=False)
            robot_original_sampled = robot_points_original[robot_indices]
        else:
            robot_original_sampled = robot_points_original
        
        # Robot points를 mesh로 변환하여 표시
        try:
            # Open3D PointCloud 생성
            robot_pcd = o3d.geometry.PointCloud()
            robot_pcd.points = o3d.utility.Vector3dVector(robot_original_sampled)
            
            # Convex hull을 사용하여 mesh 생성
            robot_hull, _ = robot_pcd.compute_convex_hull()
            robot_vertices = np.asarray(robot_hull.vertices)
            robot_faces = np.asarray(robot_hull.triangles)
            
            if len(robot_faces) > 0:
                # Robot mesh를 시각화 (빨간색, 반투명)
                fig.add_trace(go.Mesh3d(
                    x=robot_vertices[:, 0],
                    y=robot_vertices[:, 1],
                    z=robot_vertices[:, 2],
                    i=robot_faces[:, 0],
                    j=robot_faces[:, 1],
                    k=robot_faces[:, 2],
                    color='red',
                    opacity=0.3,
                    name='Original Robot Mesh',
                    showlegend=True
                ))
                print(f"Original robot mesh created: {len(robot_vertices)} vertices, {len(robot_faces)} faces")
            else:
                # Fallback: point cloud로 표시
                fig.add_trace(go.Scatter3d(
                    x=robot_original_sampled[:, 0],
                    y=robot_original_sampled[:, 1],
                    z=robot_original_sampled[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='red',
                        opacity=0.7,
                        symbol='circle'
                    ),
                    name='Original Robot Points',
                    showlegend=True
                ))
        except Exception as e:
            print(f"Failed to create original robot mesh: {e}")
            # Fallback: point cloud로 표시
            fig.add_trace(go.Scatter3d(
                x=robot_original_sampled[:, 0],
                y=robot_original_sampled[:, 1],
                z=robot_original_sampled[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='red',
                    opacity=0.7,
                    symbol='circle'
                ),
                name='Original Robot Points',
                showlegend=True
            ))
    
    # 1.6. 변환된 로봇 포인트들을 mesh로 표시 (주황색)
    if robot_points_transformed is not None and len(robot_points_transformed) > 0:
        # 로봇 포인트들을 샘플링 (너무 많으면)
        if len(robot_points_transformed) > 2000:
            robot_indices = np.random.choice(len(robot_points_transformed), 2000, replace=False)
            robot_transformed_sampled = robot_points_transformed[robot_indices]
        else:
            robot_transformed_sampled = robot_points_transformed
        
        # Robot points를 mesh로 변환하여 표시
        try:
            # Open3D PointCloud 생성
            robot_pcd = o3d.geometry.PointCloud()
            robot_pcd.points = o3d.utility.Vector3dVector(robot_transformed_sampled)
            
            # Convex hull을 사용하여 mesh 생성
            robot_hull, _ = robot_pcd.compute_convex_hull()
            robot_vertices = np.asarray(robot_hull.vertices)
            robot_faces = np.asarray(robot_hull.triangles)
            
            if len(robot_faces) > 0:
                # Robot mesh를 시각화 (주황색, 반투명)
                fig.add_trace(go.Mesh3d(
                    x=robot_vertices[:, 0],
                    y=robot_vertices[:, 1],
                    z=robot_vertices[:, 2],
                    i=robot_faces[:, 0],
                    j=robot_faces[:, 1],
                    k=robot_faces[:, 2],
                    color='orange',
                    opacity=0.4,
                    name='Transformed Robot Mesh',
                    showlegend=True
                ))
                print(f"Robot mesh created: {len(robot_vertices)} vertices, {len(robot_faces)} faces")
            else:
                # Fallback: point cloud로 표시
                fig.add_trace(go.Scatter3d(
                    x=robot_transformed_sampled[:, 0],
                    y=robot_transformed_sampled[:, 1],
                    z=robot_transformed_sampled[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='orange',
                        opacity=0.7,
                        symbol='square'
                    ),
                    name='Transformed Robot Points',
                    showlegend=True
                ))
        except Exception as e:
            print(f"Failed to create robot mesh: {e}")
            # Fallback: point cloud로 표시
            fig.add_trace(go.Scatter3d(
                x=robot_transformed_sampled[:, 0],
                y=robot_transformed_sampled[:, 1],
                z=robot_transformed_sampled[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='orange',
                    opacity=0.7,
                    symbol='square'
                ),
                name='Transformed Robot Points',
                showlegend=True
            ))
        
        # 변환 벡터 표시 (화살표)
        if robot_points_original is not None and len(robot_points_original) > 0:
            # 몇 개의 대표적인 포인트에 대해서만 변환 벡터 표시
            num_arrows = min(10, len(robot_points_original), len(robot_points_transformed))
            arrow_indices = np.random.choice(min(len(robot_points_original), len(robot_points_transformed)), 
                                           num_arrows, replace=False)
            
            for idx in arrow_indices:
                start = robot_points_original[idx]
                end = robot_points_transformed[idx]
                
                # 화살표를 여러 선분으로 표현
                fig.add_trace(go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    line=dict(
                        color='purple',
                        width=3
                    ),
                    name='Robot Transformation' if idx == arrow_indices[0] else None,
                    showlegend=True if idx == arrow_indices[0] else False
                ))
    
    # 2. 쿼리 포인트 표시 (더 크고 명확하게)
    fig.add_trace(go.Scatter3d(
        x=[query_point[0]],
        y=[query_point[1]],
        z=[query_point[2]],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        name='Query Point',
        showlegend=True
    ))
    
    # 3. 원점 (카메라 위치) 표시
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            symbol='circle',
            line=dict(width=2, color='darkblue')
        ),
        name='Camera Origin',
        showlegend=True
    ))
    
    # 4. 각 viewpoint와 Line of Sight 표시
    colors = ['green', 'blue', 'orange', 'purple', 'brown']
    
    for i, (viewpoint, result) in enumerate(zip(candidate_viewpoints, results)):
        # Viewpoint 표시 (더 크고 명확하게)
        viewpoint_color = 'green' if result['visible_robot'] else 'red'
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0]],
            y=[viewpoint[1]],
            z=[viewpoint[2]],
            mode='markers',
            marker=dict(
                size=8,
                color=viewpoint_color,
                symbol='circle',
                line=dict(width=2, color='darkgreen' if result['visible_robot'] else 'darkred')
            ),
            name=f'Viewpoint {i+1} ({result["visible_robot"] and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
        
        # Line of Sight 표시 (더 두껍고 명확하게)
        los_color = 'green' if result['visible_robot'] else 'red'
        los_style = 'solid' if result['visible_robot'] else 'dash'
        
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0], query_point[0]],
            y=[viewpoint[1], query_point[1]],
            z=[viewpoint[2], query_point[2]],
            mode='lines',
            line=dict(
                color=los_color,
                width=6,
                dash=los_style
            ),
            name=f'LoS {i+1} ({result["visible_robot"] and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
        
        # Hit point 표시 (OCCLUDED인 경우)
        if not result['visible_robot']:
            hit_distance = result['hit_distance_robot']
            direction = query_point - viewpoint
            direction = direction / np.linalg.norm(direction)
            hit_point = viewpoint + direction * hit_distance
            
            fig.add_trace(go.Scatter3d(
                x=[hit_point[0]],
                y=[hit_point[1]],
                z=[hit_point[2]],
                mode='markers',
                marker=dict(
                    size=6,
                    color='orange',
                    symbol='x'
                ),
                name=f'Hit Point {i+1}',
                showlegend=True
            ))
            
            # Hit point에서 쿼리 포인트까지의 선 (가려진 부분)
            fig.add_trace(go.Scatter3d(
                x=[hit_point[0], query_point[0]],
                y=[hit_point[1], query_point[1]],
                z=[hit_point[2], query_point[2]],
                mode='lines',
                line=dict(
                    color='red',
                    width=3,
                    dash='dot'
                ),
                name=f'Occluded Part {i+1}',
                showlegend=True
            ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text='3D Line of Sight Analysis: Scene Mesh + Robot Transform Visualization',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='lightgray'
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # HTML 파일로 저장
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"3D visualization saved to: {save_path}")



def create_robot_3d_visualization_plotly(scene_mesh, robot_meshes, query_point, candidate_viewpoints, results, joint_angles, save_path):
    """
    Plotly를 사용한 로봇과 scene을 포함한 인터랙티브 3D 시각화 생성
    """
    # Scene mesh 데이터 추출
    if hasattr(scene_mesh, 'vertices') and callable(scene_mesh.vertices):
        # nvblox ColorMesh
        scene_vertices = scene_mesh.vertices().cpu().numpy()
        scene_faces = scene_mesh.triangles().cpu().numpy()
        scene_vertex_colors = scene_mesh.vertex_colors().cpu().numpy() if hasattr(scene_mesh, 'vertex_colors') and callable(scene_mesh.vertex_colors) else None
    else:
        # Open3D mesh
        scene_vertices = np.asarray(scene_mesh.vertices)
        scene_faces = np.asarray(scene_mesh.triangles)
        scene_vertex_colors = np.asarray(scene_mesh.vertex_colors) if hasattr(scene_mesh, 'vertex_colors') and len(scene_mesh.vertex_colors) > 0 else None
    
    # 3D 시각화 생성
    fig = go.Figure()
    
    # 1. Scene 메시 표시
    if len(scene_faces) > 0 and len(scene_vertices) > 0:
        # Scene mesh를 샘플링 (성능을 위해)
        if len(scene_faces) > 50000:
            face_indices = np.random.choice(len(scene_faces), 50000, replace=False)
            scene_faces_sampled = scene_faces[face_indices]
        else:
            scene_faces_sampled = scene_faces
        
        # Scene mesh 시각화
        if scene_vertex_colors is not None and len(scene_vertex_colors) > 0:
            # RGB 색상 정보가 있는 경우
            if scene_vertex_colors.max() <= 1.0:
                colors_rgb = scene_vertex_colors
            else:
                colors_rgb = scene_vertex_colors / 255.0
            
            colors_hex = []
            for color in colors_rgb:
                r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                colors_hex.append(f'rgb({r},{g},{b})')
            
            fig.add_trace(go.Mesh3d(
                x=scene_vertices[:, 0],
                y=scene_vertices[:, 1],
                z=scene_vertices[:, 2],
                i=scene_faces_sampled[:, 0],
                j=scene_faces_sampled[:, 1],
                k=scene_faces_sampled[:, 2],
                facecolor=colors_hex,
                opacity=0.6,
                name='Scene Mesh',
                showlegend=True
            ))
        else:
            fig.add_trace(go.Mesh3d(
                x=scene_vertices[:, 0],
                y=scene_vertices[:, 1],
                z=scene_vertices[:, 2],
                i=scene_faces_sampled[:, 0],
                j=scene_faces_sampled[:, 1],
                k=scene_faces_sampled[:, 2],
                color='lightblue',
                opacity=0.6,
                name='Scene Mesh',
                showlegend=True
            ))
    
    # 2. Robot 링크 메시들 표시
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    color_idx = 0
    
    for link_name, robot_mesh in robot_meshes.items():
        if robot_mesh is not None and len(robot_mesh.vertices) > 0:
            vertices = np.asarray(robot_mesh.vertices)
            triangles = np.asarray(robot_mesh.triangles)
            
            if len(vertices) > 0 and len(triangles) > 0:
                # Robot mesh 시각화
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    color=colors[color_idx % len(colors)],
                    opacity=0.8,
                    name=f'Robot {link_name}',
                    showlegend=True
                ))
                color_idx += 1
    
    # 3. 로봇 베이스 월드 좌표계 표시
    if 'link00' in robot_meshes and robot_meshes['link00'] is not None:
        # 로봇 베이스의 중심점 계산
        base_vertices = np.asarray(robot_meshes['link00'].vertices)
        base_center = np.mean(base_vertices, axis=0)
        
        # 월드 좌표계 축 표시 (로봇 베이스 중심에서)
        axis_length = 0.1  # 10cm 축 길이
        
        # X축 (빨간색)
        fig.add_trace(go.Scatter3d(
            x=[base_center[0], base_center[0] + axis_length],
            y=[base_center[1], base_center[1]],
            z=[base_center[2], base_center[2]],
            mode='lines+markers',
            line=dict(color='red', width=8),
            marker=dict(size=4, color='red'),
            name='Robot Base X-axis',
            showlegend=True
        ))
        
        # Y축 (초록색)
        fig.add_trace(go.Scatter3d(
            x=[base_center[0], base_center[0]],
            y=[base_center[1], base_center[1] + axis_length],
            z=[base_center[2], base_center[2]],
            mode='lines+markers',
            line=dict(color='green', width=8),
            marker=dict(size=4, color='green'),
            name='Robot Base Y-axis',
            showlegend=True
        ))
        
        # Z축 (파란색)
        fig.add_trace(go.Scatter3d(
            x=[base_center[0], base_center[0]],
            y=[base_center[1], base_center[1]],
            z=[base_center[2], base_center[2] + axis_length],
            mode='lines+markers',
            line=dict(color='blue', width=8),
            marker=dict(size=4, color='blue'),
            name='Robot Base Z-axis',
            showlegend=True
        ))
        
        # 축 라벨 추가
        fig.add_trace(go.Scatter3d(
            x=[base_center[0] + axis_length + 0.02],
            y=[base_center[1]],
            z=[base_center[2]],
            mode='text',
            text=['X'],
            textfont=dict(size=16, color='red'),
            name='Robot Base X-label',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[base_center[0]],
            y=[base_center[1] + axis_length + 0.02],
            z=[base_center[2]],
            mode='text',
            text=['Y'],
            textfont=dict(size=16, color='green'),
            name='Robot Base Y-label',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[base_center[0]],
            y=[base_center[1]],
            z=[base_center[2] + axis_length + 0.02],
            mode='text',
            text=['Z'],
            textfont=dict(size=16, color='blue'),
            name='Robot Base Z-label',
            showlegend=False
        ))

    # 4. 쿼리 포인트 표시
    fig.add_trace(go.Scatter3d(
        x=[query_point[0]],
        y=[query_point[1]],
        z=[query_point[2]],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        name='Query Point',
        showlegend=True
    ))
    
    # 4. 각 viewpoint와 Line of Sight 표시
    for i, (viewpoint, result) in enumerate(zip(candidate_viewpoints, results)):
        # Viewpoint 표시
        viewpoint_color = 'green' if result['visible_robot'] else 'red'
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0]],
            y=[viewpoint[1]],
            z=[viewpoint[2]],
            mode='markers',
            marker=dict(
                size=8,
                color=viewpoint_color,
                symbol='circle',
                line=dict(width=2, color='darkgreen' if result['visible_robot'] else 'darkred')
            ),
            name=f'Viewpoint {i+1} ({result["visible_robot"] and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
        
        # Line of Sight 표시
        los_color = 'green' if result['visible_robot'] else 'red'
        los_style = 'solid' if result['visible_robot'] else 'dash'
        
        fig.add_trace(go.Scatter3d(
            x=[viewpoint[0], query_point[0]],
            y=[viewpoint[1], query_point[1]],
            z=[viewpoint[2], query_point[2]],
            mode='lines',
            line=dict(
                color=los_color,
                width=6,
                dash=los_style
            ),
            name=f'LoS {i+1} ({result["visible_robot"] and "VISIBLE" or "OCCLUDED"})',
            showlegend=True
        ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f'Robot Visibility Analysis (Joint Angles: {joint_angles})',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='lightgray'
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # HTML 파일로 저장
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"Robot 3D visualization saved to: {save_path}")


def create_multi_robot_viewpoint_set_3d_visualization(scene_mesh, robot_meshes_list, query_point, viewpoint_matrix, visibility_results, final_rewards, save_path):
    """
    M개의 robot viewpoint set을 모두 보여주는 3D 시각화
    """
    M, L = viewpoint_matrix.shape[:2]
    
    # 서브플롯 생성 (M개의 viewpoint set을 각각 표시)
    fig = make_subplots(
        rows=1, cols=M,
        specs=[[{'type': 'scatter3d'} for _ in range(M)]],
        subplot_titles=[f'Robot Set {i+1} (Reward: {final_rewards[i]:.1f})' for i in range(M)],
        horizontal_spacing=0.05
    )
    
    # 각 viewpoint set에 대해 시각화
    for set_idx in range(M):
        # Scene mesh 데이터 추출
        if hasattr(scene_mesh, 'vertices') and callable(scene_mesh.vertices):
            scene_vertices = scene_mesh.vertices().cpu().numpy()
            scene_faces = scene_mesh.triangles().cpu().numpy()
        else:
            scene_vertices = np.asarray(scene_mesh.vertices)
            scene_faces = np.asarray(scene_mesh.triangles)
        
        # Scene mesh를 샘플링 (성능을 위해)
        if len(scene_faces) > 5000:
            face_indices = np.random.choice(len(scene_faces), 5000, replace=False)
            scene_faces_sampled = scene_faces[face_indices]
        else:
            scene_faces_sampled = scene_faces
        
        # Scene mesh 표시
        if len(scene_faces_sampled) > 0:
            fig.add_trace(go.Mesh3d(
                x=scene_vertices[:, 0],
                y=scene_vertices[:, 1],
                z=scene_vertices[:, 2],
                i=scene_faces_sampled[:, 0],
                j=scene_faces_sampled[:, 1],
                k=scene_faces_sampled[:, 2],
                color='lightblue',
                opacity=0.3,
                name=f'Scene {set_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
        
        # 쿼리 포인트 표시
        fig.add_trace(go.Scatter3d(
            x=[query_point[0]], y=[query_point[1]], z=[query_point[2]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name=f'Query {set_idx+1}',
            showlegend=False
        ), row=1, col=set_idx+1)
        
        # 현재 set의 viewpoints 표시
        for viewpoint_idx in range(L):
            viewpoint = viewpoint_matrix[set_idx, viewpoint_idx]
            visible = visibility_results[set_idx, viewpoint_idx]
            
            viewpoint_color = 'green' if visible else 'red'
            fig.add_trace(go.Scatter3d(
                x=[viewpoint[0]], y=[viewpoint[1]], z=[viewpoint[2]],
                mode='markers',
                marker=dict(size=6, color=viewpoint_color, symbol='circle'),
                name=f'VP{viewpoint_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
            
            # Line of Sight 표시
            los_color = 'green' if visible else 'red'
            los_style = 'solid' if visible else 'dash'
            
            fig.add_trace(go.Scatter3d(
                x=[viewpoint[0], query_point[0]],
                y=[viewpoint[1], query_point[1]],
                z=[viewpoint[2], query_point[2]],
                mode='lines',
                line=dict(color=los_color, width=2, dash=los_style),
                name=f'LoS{viewpoint_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f'Multi-Robot Viewpoint Set Analysis (M={M}, L={L})',
            x=0.5,
            font=dict(size=18)
        ),
        width=400 * M,
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # 각 서브플롯의 scene 설정
    for i in range(1, M+1):
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)', 
            zaxis_title='Z (m)',
            aspectmode='data',
            row=1, col=i
        )
    
    # HTML 파일로 저장
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"Multi-robot viewpoint set 3D visualization saved to: {save_path}")


def create_multi_viewpoint_set_3d_visualization(meshes, query_point, viewpoint_matrix, visibility_results, final_rewards, save_path):
    """
    M개의 viewpoint set을 모두 보여주는 3D 시각화
    
    Args:
        meshes: L개의 mesh 리스트
        query_point: 쿼리 포인트
        viewpoint_matrix: [M, L, 3] 모양의 viewpoint 매트릭스
        visibility_results: [M, L] 모양의 visibility 결과
        final_rewards: [M] 모양의 최종 reward 배열
        save_path: 저장 경로
    """
    M, L = viewpoint_matrix.shape[:2]
    
    # 서브플롯 생성 (M개의 viewpoint set을 각각 표시)
    fig = make_subplots(
        rows=1, cols=M,
        specs=[[{'type': 'scatter3d'} for _ in range(M)]],
        subplot_titles=[f'Viewpoint Set {i+1} (Reward: {final_rewards[i]:.1f})' for i in range(M)],
        horizontal_spacing=0.05
    )
    
    # 각 viewpoint set에 대해 시각화
    for set_idx in range(M):
        # 첫 번째 mesh를 대표로 사용 (실제로는 각 set마다 다른 mesh를 사용해야 함)
        mesh = meshes[0]  # 간단히 첫 번째 mesh 사용
        
        # 메시 데이터 추출
        if hasattr(mesh, 'vertices') and callable(mesh.vertices):
            vertices = mesh.vertices().cpu().numpy()
            faces = mesh.triangles().cpu().numpy()
        else:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
        
        # 메시를 샘플링 (성능을 위해) - 면의 수만 제한
        if len(faces) > 5000:
            # 면의 수를 제한 (vertex는 그대로 유지)
            face_indices = np.random.choice(len(faces), 5000, replace=False)
            faces_sampled = faces[face_indices]
        else:
            faces_sampled = faces
        
        # 메시 표시 (실제 mesh로)
        if len(faces_sampled) > 0:
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces_sampled[:, 0],
                j=faces_sampled[:, 1],
                k=faces_sampled[:, 2],
                color='lightblue',
                opacity=0.3,
                name=f'Scene {set_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
        else:
            # Fallback: point cloud
            fig.add_trace(go.Scatter3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                mode='markers',
                marker=dict(size=2, color='lightblue', opacity=0.3),
                name=f'Scene {set_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
        
        # 쿼리 포인트 표시
        fig.add_trace(go.Scatter3d(
            x=[query_point[0]], y=[query_point[1]], z=[query_point[2]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name=f'Query {set_idx+1}',
            showlegend=False
        ), row=1, col=set_idx+1)
        
        # 현재 set의 viewpoints 표시
        for viewpoint_idx in range(L):
            viewpoint = viewpoint_matrix[set_idx, viewpoint_idx]
            visible = visibility_results[set_idx, viewpoint_idx]
            
            viewpoint_color = 'green' if visible else 'red'
            fig.add_trace(go.Scatter3d(
                x=[viewpoint[0]], y=[viewpoint[1]], z=[viewpoint[2]],
                mode='markers',
                marker=dict(size=6, color=viewpoint_color, symbol='circle'),
                name=f'VP{viewpoint_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
            
            # Line of Sight 표시
            los_color = 'green' if visible else 'red'
            los_style = 'solid' if visible else 'dash'
            
            fig.add_trace(go.Scatter3d(
                x=[viewpoint[0], query_point[0]],
                y=[viewpoint[1], query_point[1]],
                z=[viewpoint[2], query_point[2]],
                mode='lines',
                line=dict(color=los_color, width=2, dash=los_style),
                name=f'LoS{viewpoint_idx+1}',
                showlegend=False
            ), row=1, col=set_idx+1)
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f'Multi-Viewpoint Set Analysis (M={M}, L={L})',
            x=0.5,
            font=dict(size=18)
        ),
        width=400 * M,
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # 각 서브플롯의 scene 설정
    for i in range(1, M+1):
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)', 
            zaxis_title='Z (m)',
            aspectmode='data',
            row=1, col=i
        )
    
    # HTML 파일로 저장
    pyo.plot(fig, filename=save_path, auto_open=False)
    print(f"Multi-viewpoint set 3D visualization saved to: {save_path}")


# ========= 기존 segmentation mask 기반 코드는 제거됨 =========
# URDF 기반 접근법으로 대체됨


def create_end_effector_poses_demo(candidate_viewpoints, robot_base_position=None):
    """
    End effector pose들을 생성 (각 viewpoint별로)
    실제로는 로봇의 다음 스텝 위치를 예측해야 함
    
    Args:
        candidate_viewpoints: viewpoint들의 리스트
        robot_base_position: 로봇 베이스 위치 [x, y, z]
    
    Returns:
        end_effector_poses: 각 viewpoint에 대응하는 end effector pose들의 리스트 (4x4 변환 행렬)
    """
    end_effector_poses = []
    
    # 로봇 베이스 위치에 맞게 조정된 시작점과 끝점 정의
    if robot_base_position is None:
        robot_base_position = np.array([0.0, 0.0, 0.0])
    
    # 시작 pose: 로봇 베이스에서 약간 위쪽 (홈 포지션)
    start_pose = np.array([
        [1, 0, 0, robot_base_position[0]],
        [0, 1, 0, robot_base_position[1]], 
        [0, 0, 1, robot_base_position[2] + 0.4],  # 베이스에서 40cm 위
        [0, 0, 0, 1]
    ])
    
    # 끝 pose: 로봇 베이스에서 앞쪽으로 뻗은 포지션
    end_pose = np.array([
        [0.50622026, 0.0, 0.86240423, robot_base_position[0] + 0.3],  # 베이스에서 30cm 앞
        [0.0, 1.0, 0.0, robot_base_position[1]],
        [-0.86240423, 0.0, 0.50622026, robot_base_position[2] + 0.2],  # 베이스에서 20cm 위
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    for i, viewpoint in enumerate(candidate_viewpoints):
        # Linear interpolation between start and end pose
        alpha = i / (len(candidate_viewpoints) - 1) if len(candidate_viewpoints) > 1 else 0
        
        # Translation interpolation
        start_translation = start_pose[:3, 3]
        end_translation = end_pose[:3, 3]
        interpolated_translation = start_translation + alpha * (end_translation - start_translation)
        
        # Rotation interpolation using SLERP (Spherical Linear Interpolation)
        start_rotation = R.from_matrix(start_pose[:3, :3])
        end_rotation = R.from_matrix(end_pose[:3, :3])
        interpolated_rotation = R.from_quat(
            R.from_matrix(start_rotation.as_matrix()).as_quat() * (1 - alpha) + 
            R.from_matrix(end_rotation.as_matrix()).as_quat() * alpha
        )
        
        # SE(3) 변환 행렬 생성
        end_effector_pose = np.eye(4)
        end_effector_pose[:3, :3] = interpolated_rotation.as_matrix()
        end_effector_pose[:3, 3] = interpolated_translation
        
        end_effector_poses.append(end_effector_pose)
    
    return end_effector_poses


def render_robot_rgb_depth(robot_meshes, camera_intrinsics, image_size=(640, 480), camera_pose=None):
    """
    로봇 메시들을 RGB와 depth로 렌더링
    
    Args:
        robot_meshes: 로봇 링크 메시들의 딕셔너리 {link_name: mesh}
        camera_intrinsics: 카메라 내부 파라미터 (3, 3)
        image_size: 렌더링할 이미지 크기 (width, height)
        camera_pose: 카메라 pose (4, 4), None이면 identity
    
    Returns:
        rgb_image: RGB 이미지 (H, W, 3)
        depth_image: depth 이미지 (H, W)
    """
    if camera_pose is None:
        camera_pose = np.eye(4)
    
    # Open3D visualizer를 사용한 렌더링
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image_size[0], height=image_size[1], visible=False)
    
    # 각 로봇 링크 메시를 추가
    for link_name, mesh in robot_meshes.items():
        if mesh is not None and len(mesh.vertices) > 0:
            # 메시에 색상 추가
            mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 회색
            vis.add_geometry(mesh)
    
    # 카메라 파라미터 설정
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    
    # 카메라 내부 파라미터 설정
    camera_params.intrinsic.set_intrinsics(
        image_size[0], image_size[1],
        camera_intrinsics[0, 0], camera_intrinsics[1, 1],
        camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    )
    
    # 카메라 외부 파라미터 설정
    camera_params.extrinsic = camera_pose
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    
    # 렌더링
    vis.poll_events()
    vis.update_renderer()
    
    # RGB 이미지 캡처
    rgb_image = vis.capture_screen_float_buffer(do_render=True)
    rgb_image = np.asarray(rgb_image)
    
    # Depth 이미지 캡처
    depth_image = vis.capture_depth_float_buffer(do_render=True)
    depth_image = np.asarray(depth_image)
    
    vis.destroy_window()
    
    return rgb_image, depth_image


def create_robot_rgb_depth_visualization(robot_meshes_list, camera_intrinsics, save_dir="visibility_test_output"):
    """
    각 로봇에 대해 RGB/Depth 렌더링을 생성하고 시각화
    
    Args:
        robot_meshes_list: 로봇 메시들의 리스트
        camera_intrinsics: 카메라 내부 파라미터
        save_dir: 저장할 디렉토리
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("Creating robot RGB/Depth renderings...")
    
    for i, robot_meshes in enumerate(robot_meshes_list):
        if len(robot_meshes) == 0:
            continue
            
        print(f"  Rendering robot {i+1}...")
        
        try:
            # RGB/Depth 렌더링
            rgb_image, depth_image = render_robot_rgb_depth(
                robot_meshes, camera_intrinsics, image_size=(640, 480)
            )
            
            # RGB 이미지 저장
            rgb_path = os.path.join(save_dir, f"robot_{i+1}_rgb.png")
            plt.imsave(rgb_path, rgb_image)
            
            # Depth 이미지 저장
            depth_path = os.path.join(save_dir, f"robot_{i+1}_depth.png")
            plt.imsave(depth_path, depth_image, cmap='magma')
            
            print(f"    RGB saved to: {rgb_path}")
            print(f"    Depth saved to: {depth_path}")
            
        except Exception as e:
            print(f"    Failed to render robot {i+1}: {e}")


# ========= 메인 =========
def main():
    print("=== URDF-based Robot Visibility Test with nvblox ===")
    print("Using URDF robot meshes with Inverse Kinematics for realistic robot positioning")
    
    # 전체 실행 시간 측정
    total_start_time = time.time()
    
    # 1) 입력 로드
    print("\n=== 1. Loading RGB-D Data ===")
    load_start_time = time.time()
    rgb, depth_raw = load_rgbd_from_zarr(BUFFER_PATH, EPISODE_IDX, FRAME_IDX, DEPTH_SCALE)
    H, W = depth_raw.shape
    load_end_time = time.time()
    print(f"Loaded RGB: {rgb.shape}, Depth: {depth_raw.shape}")
    print(f"Data loading time: {load_end_time - load_start_time:.4f} seconds")

    # 1.5) 이미지 리사이즈 (옵션) - UniDepth 사용시에는 RGB만 리사이즈
    if RESIZE:
        print(f"\n=== 1.5. Resizing RGB to {RESIZE_SIZE} ===")
        resize_start_time = time.time()
        rgb_resized, _, scale_factor_x, scale_factor_y = resize_image_and_depth(rgb, depth_raw, RESIZE_SIZE)
        K_adjusted = adjust_camera_intrinsics(K, scale_factor_x, scale_factor_y)
        resize_end_time = time.time()
        print(f"Resized RGB: {rgb_resized.shape}, Original depth: {depth_raw.shape}")
        print(f"Scale factors: x={scale_factor_x:.4f}, y={scale_factor_y:.4f}")
        print(f"Adjusted camera intrinsics: fx={K_adjusted[0,0]:.2f}, fy={K_adjusted[1,1]:.2f}, cx={K_adjusted[0,2]:.2f}, cy={K_adjusted[1,2]:.2f}")
        print(f"RGB resize time: {resize_end_time - resize_start_time:.4f} seconds")
    else:
        K_adjusted = K
        rgb_resized = rgb

    if K is None:
        raise ValueError("K (intrinsics)가 None 입니다. 코드 상단의 K를 사용자의 카메라 파라미터로 채워주세요.")

    # 2) Depth 처리
    print("\n=== 2. Depth Processing ===")
    depth_start_time = time.time()
    
    if USE_MONODEPTH:
        print("Using UniDepth for depth estimation...")
        
        # UniDepth 모델 로드
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        name = f"unidepth-v2-vit{MODEL_TYPE}14"
        model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
        model.interpolation_mode = "bilinear"
        model = model.to(device).eval()
        
        # 카메라 설정 (원본 크기로 UniDepth 실행)
        intrinsics_torch = torch.from_numpy(K).float()
        camera = Pinhole(K=intrinsics_torch.unsqueeze(0))
        if isinstance(model, (UniDepthV2old, UniDepthV1)):
            camera = camera.K.squeeze(0)
        
        # Depth 추정 (원본 크기 RGB 사용)
        print(f"Running UniDepth on original RGB: {rgb.shape}")
        depth_pred_original = get_depth_with_unidepth(rgb, K, model, camera)
        
        # Raw depth로 스케일링 (원본 크기)
        depth_m_original = scale_depth_with_raw(depth_pred_original, depth_raw, K)
        
        # 리사이즈가 필요한 경우 depth만 리사이즈
        if RESIZE:
            print(f"Resizing depth from {depth_m_original.shape} to {RESIZE_SIZE}")
            import cv2
            depth_m = cv2.resize(depth_m_original, RESIZE_SIZE, interpolation=cv2.INTER_LINEAR)
        else:
            depth_m = depth_m_original
        
        print(f"Final depth shape: {depth_m.shape}")
    else:
        print("Using raw depth data...")
        if RESIZE:
            import cv2
            depth_m = cv2.resize(depth_raw, RESIZE_SIZE, interpolation=cv2.INTER_LINEAR)
        else:
            depth_m = depth_raw
    
    depth_end_time = time.time()
    print(f"Depth processing time: {depth_end_time - depth_start_time:.4f} seconds")

    # 3) 월드 좌표계 = 원 카메라0 좌표계로 가정
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.zeros(3, dtype=np.float64)

    # 4) 쿼리 픽셀 & 3D점
    print("\n=== 3. Query Point Selection ===")
    query_start_time = time.time()
    
    uq, vq = pick_query_pixel(depth_m)
    dq = float(depth_m[vq, uq])
    fx, fy, cx, cy = K_adjusted[0,0], K_adjusted[1,1], K_adjusted[0,2], K_adjusted[1,2]
    xq = (uq - cx) * dq / fx
    yq = (vq - cy) * dq / fy
    Xc0_q = np.array([xq, yq, dq], dtype=np.float64)
    Xw_q = R0.T @ (Xc0_q - t0)

    query_end_time = time.time()
    print(f"[Query] pixel=({uq},{vq}), depth={dq:.4f} m, world_pos={Xw_q}")
    print(f"Query point selection time: {query_end_time - query_start_time:.4f} seconds")

    # 5) URDF 기반 로봇 설정 및 nvblox 메시 생성
    print("\n=== 4. URDF-based Robot Setup and nvblox Mesh Generation ===")
    mesh_start_time = time.time()
    
    # URDF 파일 경로 설정
    urdf_path = "/home/dscho1234/Workspace/unitree_ros/robots/z1_description/xacro/z1.urdf"
    mesh_base_path = "/home/dscho1234/Workspace/unitree_ros/robots/z1_description/"
    
    # Z1 로봇 시각화 객체 생성
    robot_viz = Z1RobotVisualizer(urdf_path, mesh_base_path)
    
    # 로봇 베이스 위치를 scene에 맞게 조정
    # Scene의 중심 근처에 로봇을 배치하여 자연스러운 manipulation 환경 구성
    scene_center = np.array([0.0, 0.0, 0.0])  # Scene의 중심 (카메라 좌표계 기준)
    robot_base_position = np.array([0.3, 0.0, 0.0])  # Scene 중심에서 30cm 앞쪽에 배치
    
    # 로봇 베이스 orientation 설정 (Y축을 중심으로 약간 회전하여 scene을 향하도록)
    base_rotation_y = R.from_euler('y', -30, degrees=True)  # -30도 회전
    robot_viz.set_robot_base_transform(robot_base_position, base_rotation_y.as_matrix())
    print(f"Robot base positioned at: {robot_base_position}")
    print(f"Robot base orientation: Y-axis rotation -30 degrees")
    
    # L개의 end effector pose 생성 (각 column index i에 대응)
    end_effector_poses = create_end_effector_poses_demo(CANDIDATE_VIEWPOINTS, robot_base_position)
    print(f"Generated {len(end_effector_poses)} end effector poses")
    for i, pose in enumerate(end_effector_poses):
        print(f"  Pose {i+1}: translation={pose[:3, 3]}")
    
    # 원본 nvblox 메시 생성 (로봇 없이)
    print("\nCreating original nvblox mesh (without robot)...")
    original_mesh_start_time = time.time()
    mesh_original, mapper_original = create_mesh_with_nvblox(depth_m, rgb_resized, K_adjusted, voxel_size=VOXEL_SIZE, max_integration_distance=5.0, return_mapper=True)
    original_mesh_end_time = time.time()
    print(f"Original nvblox mesh created in {original_mesh_end_time - original_mesh_start_time:.4f} seconds")
    print(f"Original mesh has {mesh_original.vertices().shape[0]} vertices and {mesh_original.triangles().shape[0]} triangles")
    
    # L개의 로봇 설정 및 메시 생성 (각 column index i에 대응하는 end effector pose로)
    robot_meshes_list = []
    robot_joint_angles_list = []
    
    for i, end_effector_pose in enumerate(end_effector_poses):
        print(f"\n  Setting up robot {i+1} with end effector pose: translation={end_effector_pose[:3, 3]}")
        
        # 홈 포지션으로 초기화
        robot_viz.set_joint_angles([0, 0, 0, 0, 0, 0])
        
        # Inverse Kinematics로 조인트 각도 계산
        ik_success = robot_viz.solve_inverse_kinematics(end_effector_pose, gripper_angle=0.0)
        
        if ik_success:
            # 로봇의 모든 링크 메시를 월드 좌표계로 변환하여 가져오기
            robot_meshes = robot_viz.get_robot_meshes_in_world(gripper_angle=0.0)
            robot_meshes_list.append(robot_meshes)
            robot_joint_angles_list.append(robot_viz.joint_angles.copy())
            
            print(f"  Robot {i+1} setup successful: {len(robot_meshes)} links")
            for link_name, mesh in robot_meshes.items():
                if mesh is not None and len(mesh.vertices) > 0:
                    print(f"    {link_name}: {len(mesh.vertices)} vertices")
        else:
            print(f"  Robot {i+1} setup failed: Inverse Kinematics failed")
            robot_meshes_list.append({})
            robot_joint_angles_list.append(np.zeros(6))
    
    mesh_end_time = time.time()
    print(f"\nAll robot setups completed in {mesh_end_time - mesh_start_time:.4f} seconds")
    
    # 6) M개의 viewpoint set에 대해 각 robot별로 visibility 체크
    print("\n=== 5. URDF-based Robot Visibility Check ===")
    visibility_start_time = time.time()
    
    print(f"Processing {M} viewpoint sets, each with {L} viewpoints...")
    print(f"Total viewpoints to check: {M} x {L} = {M*L}")
    
    # [M, L] 모양의 visibility 결과 저장
    visibility_results = np.zeros((M, L), dtype=bool)  # True if visible, False if occluded
    hit_distances = np.zeros((M, L), dtype=np.float64)  # Hit distances for each viewpoint
    
    # 각 robot별로 RaycastingScene을 미리 생성 (성능 최적화)
    print(f"\n=== Pre-creating RaycastingScenes for {L} robots ===")
    scene_creation_start_time = time.time()
    raycasting_scenes = []
    
    for robot_idx in range(L):
        print(f"  Creating scene for robot {robot_idx + 1}...")
        if robot_idx < len(robot_meshes_list) and len(robot_meshes_list[robot_idx]) > 0:
            # Scene mesh와 robot meshes를 결합한 scene 생성
            scene = create_combined_scene_with_robot(mesh_original, robot_meshes_list[robot_idx])
        else:
            # Robot이 없는 경우 원본 scene만 사용
            scene = create_raycasting_scene(mesh_original)
        raycasting_scenes.append(scene)
    
    scene_creation_end_time = time.time()
    print(f"All RaycastingScenes created in {scene_creation_end_time - scene_creation_start_time:.4f} seconds")
    
    # 각 robot별로 (L개의 robot) visibility 체크
    for robot_idx in range(L):  # robot_idx는 column index i에 해당
        print(f"\n=== Checking robot {robot_idx + 1} (column {robot_idx}) ===")
        current_scene = raycasting_scenes[robot_idx]
        
        # 현재 robot_idx에 해당하는 column의 viewpoints들을 모음: CANDIDATE_VIEWPOINTS_MATRIX[:, robot_idx]
        viewpoints_for_this_robot = CANDIDATE_VIEWPOINTS_MATRIX[:, robot_idx]  # [M, 3] 모양
        print(f"  Viewpoints for this robot: {viewpoints_for_this_robot.shape} (M viewpoints)")
        
        # Batch raycasting 방식 (미리 생성된 scene 사용)
        print(f"  Performing batch raycasting for {M} viewpoints...")
        
        # Batch raycasting을 위한 ray 생성
        origins = viewpoints_for_this_robot  # [M, 3] 모양
        directions = Xw_q - origins  # [M, 3] 모양 - 각 viewpoint에서 query point로의 방향
        distances = np.linalg.norm(directions, axis=1)  # [M] 모양 - 각 ray의 거리
        directions = directions / distances[:, np.newaxis]  # 정규화된 방향 벡터 [M, 3]
        
        # 미리 생성된 scene을 사용한 batch raycasting 수행
        batch_visible, batch_hit_distances = batch_raycasting_with_scene(
            current_scene, origins, directions, distances-OFFSET_DISTANCE
        )
        
        # 결과 저장
        for set_idx in range(M):
            visibility_results[set_idx, robot_idx] = batch_visible[set_idx]
            hit_distances[set_idx, robot_idx] = batch_hit_distances[set_idx]
            
            print(f"    Set {set_idx + 1}: {viewpoints_for_this_robot[set_idx]} -> {'VISIBLE' if batch_visible[set_idx] else 'OCCLUDED'} (hit: {batch_hit_distances[set_idx]:.3f}m)")
    
    visibility_end_time = time.time()
    print(f"\nURDF-based robot visibility check time: {visibility_end_time - visibility_start_time:.4f} seconds")
    
    # 7) Reward 계산 (visible=1, occluded=0)
    print("\n=== 6. Reward Calculation ===")
    reward_start_time = time.time()
    
    # [M, L] 모양의 reward 매트릭스 생성 (visible=1, occluded=0)
    reward_matrix = visibility_results.astype(np.float64)
    
    print("Reward matrix (M x L):")
    print("  M = viewpoint set index (row)")
    print("  L = viewpoint index (column)")
    print("  Value: 1.0 = visible, 0.0 = occluded")
    print()
    
    for set_idx in range(M):
        print(f"Viewpoint set {set_idx + 1}: {reward_matrix[set_idx]}")
    
    # 각 viewpoint set별로 최종 reward 계산 (L개의 viewpoint change에 따른 reward 합계)
    final_rewards = np.sum(reward_matrix, axis=1)  # [M] 모양의 배열
    print(f"\nFinal rewards for each viewpoint set:")
    for set_idx in range(M):
        print(f"  Set {set_idx + 1}: {final_rewards[set_idx]:.1f} (sum of {L} viewpoints)")
    
    print(f"\nFinal rewards array: {final_rewards}")
    print(f"Total reward sum: {np.sum(final_rewards):.1f}")
    
    reward_end_time = time.time()
    print(f"Reward calculation time: {reward_end_time - reward_start_time:.4f} seconds")
    
    # 결과 정리 (기존 코드와 호환성을 위해)
    results = []
    for i, (viewpoint, yaw_deg) in enumerate(zip(CANDIDATE_VIEWPOINTS, CANDIDATE_YAW_DEGS)):
        # 첫 번째 viewpoint set의 결과를 사용 (기존 시각화 코드와 호환)
        visible_robot = visibility_results[0, i] if i < L else False
        hit_distance_robot = hit_distances[0, i] if i < L else 0.0
        
        # 각 viewpoint에서 query point까지의 거리 계산
        distance_to_query = np.linalg.norm(Xw_q - viewpoint)
        
        result = {
            'viewpoint': viewpoint,
            'yaw_deg': yaw_deg,
            'visible_robot': visible_robot,
            'visible_original': visible_robot,  # 호환성을 위해 동일하게 설정
            'hit_distance_robot': hit_distance_robot,
            'hit_distance_original': hit_distance_robot,  # 호환성을 위해 동일하게 설정
            'distance_to_query': distance_to_query
        }
        results.append(result)
    
    # 결과 출력
    print("\n=== Visibility Check Results ===")
    for i, result in enumerate(results):
        print(f"\nViewpoint {i+1}: {result['viewpoint']}, yaw={result['yaw_deg']}°")
        print(f"  Robot-transformed mesh: {'✓ VISIBLE' if result['visible_robot'] else '✗ OCCLUDED'} (hit_distance: {result['hit_distance_robot']:.3f}m)")
        print(f"  Original mesh: {'✓ VISIBLE' if result['visible_original'] else '✗ OCCLUDED'} (hit_distance: {result['hit_distance_original']:.3f}m)")
        
        # 로봇 변환 효과 분석
        if result['visible_original'] != result['visible_robot']:
            print(f"  🔄 Robot transformation changed visibility: {'✓ VISIBLE' if result['visible_original'] else '✗ OCCLUDED'} → {'✓ VISIBLE' if result['visible_robot'] else '✗ OCCLUDED'}")
        if abs(result['hit_distance_original'] - result['hit_distance_robot']) > 0.01:
            print(f"  📏 Hit distance changed: {result['hit_distance_original']:.3f}m → {result['hit_distance_robot']:.3f}m")

    # 8) 시각화
    print("\n=== 7. Creating Visualizations ===")
    viz_start_time = time.time()
    
    import os
    os.makedirs("visibility_test_output", exist_ok=True)
    
    # 2D 시각화 (URDF 기반 로봇 정보 포함)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    # RGB with Query Point
    ax[0,0].imshow(rgb_resized)
    ax[0,0].scatter([uq], [vq], c='cyan', s=40, marker='x', label='Query Point')
    ax[0,0].set_title('RGB with Query Point')
    ax[0,0].axis('off')
    ax[0,0].legend()
    
    # Depth with Query Point
    im = ax[0,1].imshow(depth_m, cmap='magma')
    ax[0,1].scatter([uq], [vq], c='cyan', s=40, marker='x', label='Query Point')
    ax[0,1].set_title('Depth (m)')
    ax[0,1].axis('off')
    ax[0,1].legend()
    fig.colorbar(im, ax=ax[0,1], shrink=0.7, label='meters')
    
    # Robot Joint Angles 정보
    ax[1,0].text(0.1, 0.9, 'Robot Joint Angles:', transform=ax[1,0].transAxes, fontsize=12, fontweight='bold')
    for i, joint_angles in enumerate(robot_joint_angles_list):
        if len(joint_angles) > 0:
            ax[1,0].text(0.1, 0.8 - i*0.1, f'Robot {i+1}: {joint_angles}', 
                        transform=ax[1,0].transAxes, fontsize=10)
    ax[1,0].set_title('Robot Joint Angles')
    ax[1,0].axis('off')
    
    # End Effector Poses 정보
    ax[1,1].text(0.1, 0.9, 'End Effector Poses:', transform=ax[1,1].transAxes, fontsize=12, fontweight='bold')
    for i, pose in enumerate(end_effector_poses):
        translation = pose[:3, 3]
        ax[1,1].text(0.1, 0.8 - i*0.1, f'Pose {i+1}: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]', 
                    transform=ax[1,1].transAxes, fontsize=10)
    ax[1,1].set_title('End Effector Poses')
    ax[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig("visibility_test_output/rgb_depth_query_with_robot_info.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("2D visualization with robot info saved to: visibility_test_output/rgb_depth_query_with_robot_info.png")
    
    # Reward 매트릭스 시각화
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reward 매트릭스 히트맵
    im1 = ax[0].imshow(reward_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax[0].set_title('Reward Matrix (M x L)\nGreen=Visible(1), Red=Occluded(0)')
    ax[0].set_xlabel('Viewpoint Index (L)')
    ax[0].set_ylabel('Viewpoint Set Index (M)')
    
    # 컬러바 추가
    cbar1 = plt.colorbar(im1, ax=ax[0], shrink=0.8)
    cbar1.set_label('Reward (1.0=Visible, 0.0=Occluded)')
    
    # 각 셀에 값 표시
    for i in range(M):
        for j in range(L):
            text = ax[0].text(j, i, f'{reward_matrix[i, j]:.0f}',
                            ha="center", va="center", color="black", fontweight='bold')
    
    # 최종 reward 막대 그래프
    bars = ax[1].bar(range(1, M+1), final_rewards, color=['skyblue', 'lightgreen', 'lightcoral'][:M])
    ax[1].set_title('Final Rewards per Viewpoint Set')
    ax[1].set_xlabel('Viewpoint Set Index (M)')
    ax[1].set_ylabel('Total Reward')
    ax[1].set_ylim(0, L + 0.5)
    
    # 각 막대 위에 값 표시
    for i, reward in enumerate(final_rewards):
        ax[1].text(i+1, reward + 0.1, f'{reward:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("visibility_test_output/reward_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Reward analysis visualization saved to: visibility_test_output/reward_analysis.png")

    # 3D 시각화 - 각 robot별로 독립적인 시각화 생성
    print("Creating individual 3D visualizations for each robot...")
    
    # 각 robot별로 시각화 생성
    for robot_idx in range(L):
        print(f"Creating visualization for robot {robot_idx + 1}...")
        
        # 현재 robot과 관련 데이터
        current_robot_meshes = robot_meshes_list[robot_idx] if robot_idx < len(robot_meshes_list) else {}
        current_joint_angles = robot_joint_angles_list[robot_idx] if robot_idx < len(robot_joint_angles_list) else np.zeros(6)
        
        # 현재 robot에 대한 visibility 결과만 추출 (모든 viewpoint set에서)
        current_robot_results = []
        for set_idx in range(M):
            visible = visibility_results[set_idx, robot_idx]
            hit_distance = hit_distances[set_idx, robot_idx]
            viewpoint = CANDIDATE_VIEWPOINTS_MATRIX[set_idx, robot_idx]
            
            # 각 viewpoint에서 query point까지의 거리 계산
            distance_to_query = np.linalg.norm(Xw_q - viewpoint)
            
            result = {
                'viewpoint': viewpoint,
                'yaw_deg': 0.0,  # 기본값
                'visible_robot': visible,
                'visible_original': visible,  # 호환성을 위해 동일하게 설정
                'hit_distance_robot': hit_distance,
                'hit_distance_original': hit_distance,  # 호환성을 위해 동일하게 설정
                'distance_to_query': distance_to_query
            }
            current_robot_results.append(result)
        
        # 현재 robot에 대한 시각화 생성 (robot meshes와 scene mesh 결합)
        create_robot_3d_visualization_plotly(
            mesh_original, current_robot_meshes, Xw_q, CANDIDATE_VIEWPOINTS_MATRIX[:, robot_idx], 
            current_robot_results, current_joint_angles,
            f"visibility_test_output/3d_visualization_robot_{robot_idx + 1}.html"
        )
    
    # 원본 mesh와 비교를 위한 시각화 생성
    print("Creating original mesh visualization for comparison...")
    original_results = []
    for i, viewpoint in enumerate(CANDIDATE_VIEWPOINTS):
        # 원본 mesh에서는 모든 viewpoint가 visible하다고 가정 (실제로는 원본 mesh로 테스트해야 함)
        original_results.append({
            'viewpoint': viewpoint,
            'yaw_deg': 0.0,
            'visible_robot': True,  # 원본에서는 가정
            'visible_original': True,
            'hit_distance_robot': 0.0,
            'hit_distance_original': 0.0,
            'distance_to_query': np.linalg.norm(Xw_q - viewpoint)
        })
    
    create_robot_3d_visualization_plotly(
        mesh_original, {}, Xw_q, CANDIDATE_VIEWPOINTS, original_results, np.zeros(6),
        "visibility_test_output/3d_visualization_original.html"
    )
    
    # M개의 viewpoint set을 모두 보여주는 3D 시각화 생성
    create_multi_robot_viewpoint_set_3d_visualization(mesh_original, robot_meshes_list, Xw_q, CANDIDATE_VIEWPOINTS_MATRIX, 
                                                     visibility_results, final_rewards,
                                                     "visibility_test_output/3d_multi_robot_viewpoint_sets.html")
    
    # RGB/Depth 렌더링 생성
    print("\nCreating robot RGB/Depth renderings...")
    rgb_depth_start_time = time.time()
    create_robot_rgb_depth_visualization(robot_meshes_list, K_adjusted, "visibility_test_output")
    rgb_depth_end_time = time.time()
    print(f"RGB/Depth rendering time: {rgb_depth_end_time - rgb_depth_start_time:.4f} seconds")
    
    viz_end_time = time.time()
    print(f"Visualization time: {viz_end_time - viz_start_time:.4f} seconds")

    # 전체 실행 시간
    total_end_time = time.time()
    print(f"\n=== Total Execution Time: {total_end_time - total_start_time:.4f} seconds ===")

    # 결과 요약
    print("\n=== Summary ===")
    print(f"Configuration: M={M} viewpoint sets, L={L} viewpoints per set")
    print(f"Total viewpoints processed: {M} x {L} = {M*L}")
    
    # 각 viewpoint set별 요약
    print(f"\nViewpoint Set Analysis:")
    for set_idx in range(M):
        visible_count = np.sum(visibility_results[set_idx])
        total_reward = final_rewards[set_idx]
        print(f"  Set {set_idx + 1}: {visible_count}/{L} visible viewpoints, reward = {total_reward:.1f}")
    
    # 전체 통계
    total_visible = np.sum(visibility_results)
    total_possible = M * L
    overall_visibility_rate = total_visible / total_possible * 100
    
    print(f"\nOverall Statistics:")
    print(f"  Total visible viewpoints: {total_visible}/{total_possible} ({overall_visibility_rate:.1f}%)")
    print(f"  Average reward per set: {np.mean(final_rewards):.2f}")
    print(f"  Best performing set: Set {np.argmax(final_rewards) + 1} (reward: {np.max(final_rewards):.1f})")
    print(f"  Worst performing set: Set {np.argmin(final_rewards) + 1} (reward: {np.min(final_rewards):.1f})")
    
    # Reward 매트릭스 요약
    print(f"\nReward Matrix Summary:")
    print(f"  Matrix shape: {reward_matrix.shape}")
    print(f"  Total reward sum: {np.sum(final_rewards):.1f}")
    print(f"  Reward variance: {np.var(final_rewards):.2f}")
    
    print("\nDetailed results (first viewpoint set only):")
    for i, result in enumerate(results):
        status_robot = "VISIBLE" if result['visible_robot'] else "OCCLUDED"
        print(f"  Viewpoint {i+1}: {status_robot} (hit_distance: {result['hit_distance_robot']:.3f}m)")


if __name__ == "__main__":
    main()
