"""
CT预处理模块

该模块提供3D医学CT数据的预处理功能，包括：
- 分辨率适配（向上兼容到512³或1024³）
- 窗宽/窗位二值化
- 器官特定窗口处理
- 语义分割标签处理
"""

from .config import (
    WINDOW_CONFIGS,
    AIR_HU_VALUE,
    DEFAULT_RESOLUTION,
    SUPPORTED_RESOLUTIONS,
    get_window_config,
    add_custom_window
)

from .resolution_adapter import (
    determine_target_resolution,
    adapt_resolution,
    check_resolution_compatibility
)

from .window_processor import (
    apply_window_binarization,
    process_all_windows,
    get_window_filename
)

from .organ_extractor import (
    extract_organ_with_window,
    process_all_organs,
    load_organ_mapping,
    get_organs_present,
    validate_segmentation,
    compute_organ_statistics
)

__all__ = [
    # Config
    'WINDOW_CONFIGS',
    'AIR_HU_VALUE',
    'DEFAULT_RESOLUTION',
    'SUPPORTED_RESOLUTIONS',
    'get_window_config',
    'add_custom_window',
    
    # Resolution adapter
    'determine_target_resolution',
    'adapt_resolution',
    'check_resolution_compatibility',
    
    # Window processor
    'apply_window_binarization',
    'process_all_windows',
    'get_window_filename',
    
    # Organ extractor
    'extract_organ_with_window',
    'process_all_organs',
    'load_organ_mapping',
    'get_organs_present',
    'validate_segmentation',
    'compute_organ_statistics',
]

__version__ = '1.0.0'

