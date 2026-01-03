"""
CT窗口配置模块

定义标准的CT窗宽/窗位设置，用于不同组织类型的可视化和分析。
"""

from typing import Dict, List, Any
import copy

# 标准窗宽/窗位配置
# 窗宽 (Window Width): 显示的HU值范围
# 窗位 (Window Level): 窗口的中心HU值
WINDOW_CONFIGS = {
    'lung': {
        'window_width': 1500,
        'window_level': -600,
        'hu_min': -1350,  # level - width/2
        'hu_max': 150,    # level + width/2
        'organ_types': ['lung', 'bronchus', 'trachea'],
        'description': '肺窗 - 用于显示肺实质、支气管和微小结节'
    },
    'bone': {
        'window_width': 1500,
        'window_level': 300,
        'hu_min': -450,
        'hu_max': 1050,
        'organ_types': ['bone', 'vertebra', 'rib', 'skull', 'femur'],
        'description': '骨窗 - 用于显示骨组织结构'
    },
    'soft_tissue': {
        'window_width': 400,
        'window_level': 50,
        'hu_min': -150,
        'hu_max': 250,
        'organ_types': ['liver', 'kidney', 'spleen', 'pancreas', 'heart', 
                       'left_kidney', 'right_kidney', 'left kidney', 'right kidney'],
        'description': '软组织窗 - 用于显示腹部器官和软组织'
    },
    'brain': {
        'window_width': 80,
        'window_level': 35,
        'hu_min': -5,
        'hu_max': 75,
        'organ_types': ['brain', 'grey_matter', 'white_matter', 'cerebellum'],
        'description': '脑窗 - 用于显示脑组织结构'
    }
}

# 空气的HU值（用于填充）
AIR_HU_VALUE = -2000

# 默认目标分辨率
DEFAULT_RESOLUTION = 512

# 支持的分辨率列表
SUPPORTED_RESOLUTIONS = [512, 1024]

# 器官名称标准化映射（处理不同命名方式）
ORGAN_NAME_MAPPING = {
    'left kidney': 'left_kidney',
    'right kidney': 'right_kidney',
    'left_kidney': 'left_kidney',
    'right_kidney': 'right_kidney',
    'kidney': 'kidney',
}


def get_window_config(window_name: str) -> Dict[str, Any]:
    """
    获取指定窗口的配置
    
    Args:
        window_name: 窗口名称 ('lung', 'bone', 'soft_tissue', 'brain')
    
    Returns:
        窗口配置字典
    
    Raises:
        ValueError: 如果窗口名称不存在
    """
    if window_name not in WINDOW_CONFIGS:
        available = ', '.join(WINDOW_CONFIGS.keys())
        raise ValueError(f"Unknown window name: {window_name}. Available: {available}")
    
    return copy.deepcopy(WINDOW_CONFIGS[window_name])


def add_custom_window(window_name: str, 
                     window_width: float, 
                     window_level: float,
                     organ_types: List[str] = None,
                     description: str = '') -> None:
    """
    添加自定义窗口配置
    
    Args:
        window_name: 窗口名称
        window_width: 窗宽
        window_level: 窗位
        organ_types: 适用的器官类型列表
        description: 窗口描述
    """
    if window_name in WINDOW_CONFIGS:
        print(f"Warning: Overwriting existing window config: {window_name}")
    
    hu_min = window_level - window_width / 2
    hu_max = window_level + window_width / 2
    
    WINDOW_CONFIGS[window_name] = {
        'window_width': window_width,
        'window_level': window_level,
        'hu_min': hu_min,
        'hu_max': hu_max,
        'organ_types': organ_types or [],
        'description': description
    }


def get_window_for_organ(organ_name: str) -> str:
    """
    根据器官名称自动确定适用的窗口
    
    Args:
        organ_name: 器官名称
    
    Returns:
        窗口名称
    """
    # 标准化器官名称
    organ_name_lower = organ_name.lower().strip()
    organ_name_normalized = ORGAN_NAME_MAPPING.get(organ_name_lower, organ_name_lower)
    
    # 遍历所有窗口配置，查找匹配的器官类型
    for window_name, config in WINDOW_CONFIGS.items():
        organ_types_lower = [ot.lower() for ot in config['organ_types']]
        if organ_name_normalized in organ_types_lower or organ_name_lower in organ_types_lower:
            return window_name
    
    # 默认返回软组织窗
    return 'soft_tissue'


def validate_resolution(resolution: int) -> bool:
    """
    验证分辨率是否支持
    
    Args:
        resolution: 目标分辨率
    
    Returns:
        True if supported, False otherwise
    """
    return resolution in SUPPORTED_RESOLUTIONS


def get_all_window_names() -> List[str]:
    """
    获取所有可用的窗口名称
    
    Returns:
        窗口名称列表
    """
    return list(WINDOW_CONFIGS.keys())


if __name__ == '__main__':
    # 测试配置
    print("=== CT窗口配置 ===")
    for window_name, config in WINDOW_CONFIGS.items():
        print(f"\n{window_name.upper()}:")
        print(f"  窗宽: {config['window_width']} HU")
        print(f"  窗位: {config['window_level']} HU")
        print(f"  HU范围: [{config['hu_min']}, {config['hu_max']}]")
        print(f"  适用器官: {', '.join(config['organ_types'])}")
        print(f"  说明: {config['description']}")
    
    print(f"\n默认分辨率: {DEFAULT_RESOLUTION}")
    print(f"支持的分辨率: {SUPPORTED_RESOLUTIONS}")
    print(f"空气HU值: {AIR_HU_VALUE}")

