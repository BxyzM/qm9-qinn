"""
General configuration loader for YAML files with nested attribute access.

Provides a flexible Config class that converts nested dictionaries from YAML
into objects supporting dot notation access (e.g., cfg.training.batch_size).

Author: Dr. Aritra Bal (ETP)
Date: December 04, 2025
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Union
from loguru import logger


class Config:
    """
    Configuration object with nested attribute access.
    
    Recursively converts nested dictionaries into Config objects,
    allowing access via dot notation (cfg.section.parameter) while
    maintaining dictionary-style access (cfg['section']['parameter']).
    
    Args:
        config_dict: Dictionary to convert to Config object
    """
    
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """Initialize Config from dictionary."""
        for key, value in config_dict.items():
            setattr(self, key, self._convert(value))
    
    def _convert(self, value: Any) -> Any:
        """
        Recursively convert dictionaries to Config objects.
        
        Args:
            value: Value to convert
            
        Returns:
            Config object if value is dict, otherwise original value
        """
        if isinstance(value, dict):
            return Config(value)
        elif isinstance(value, list):
            return [self._convert(item) for item in value]
        else:
            return value
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment."""
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return hasattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get attribute with default value.
        
        Args:
            key: Attribute name
            default: Default value if attribute doesn't exist
            
        Returns:
            Attribute value or default
        """
        return getattr(self, key, default)
    
    def __repr__(self) -> str:
        """String representation of Config."""
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                items.append(f"{key}=<Config>")
            else:
                items.append(f"{key}={repr(value)}")
        return f"Config({', '.join(items)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config back to nested dictionary.
        
        Returns:
            Dictionary representation of Config
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update config with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        for key, value in updates.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Config) and isinstance(value, dict):
                getattr(self, key).update(value)
            else:
                setattr(self, key, self._convert(value))
    
    def keys(self):
        """Return config keys."""
        return self.__dict__.keys()
    
    def values(self):
        """Return config values."""
        return self.__dict__.values()
    
    def items(self):
        """Return config items."""
        return self.__dict__.items()


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with nested attribute access
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            logger.error(f"Empty configuration file: {config_path}")
            raise ValueError(f"Empty configuration file: {config_path}")
        
        config = Config(config_dict)
        logger.info("Configuration loaded successfully")
        
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise


def save_config(config: Config, output_path: Union[str, Path]) -> None:
    """
    Save Config object to YAML file.
    
    Args:
        config: Config object to save
        output_path: Path where YAML file will be saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise


def merge_configs(base_config: Config, override_config: Config) -> Config:
    """
    Merge two configs, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration with override values
        
    Returns:
        Merged Config object
    """
    merged_dict = base_config.to_dict()
    override_dict = override_config.to_dict()
    
    def deep_update(base: Dict, override: Dict) -> Dict:
        """Recursively update nested dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    merged_dict = deep_update(merged_dict, override_dict)
    return Config(merged_dict)


# Testing block
if __name__ == "__main__":
    import tempfile
    import argparse
    
    parser = argparse.ArgumentParser(description="Test config loader")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If not provided, creates test config."
    )
    args = parser.parse_args()
    
    if args.config is None:
        logger.info("Creating test configuration")
        
        # Create test config
        test_config_dict = {
            'circuit': {
                'qubits': 5,
                'layers': 3,
                'backend': 'autograd'
            },
            'training': {
                'epochs': 50,
                'batch_size': 1,
                'lr_decay': True
            },
            'data': {
                'protein_pairs': ['1TM1_E_I', '3FK9_A_B'],
                'shuffle': True
            }
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config_dict, f)
            temp_path = f.name
        
        config_path = temp_path
        logger.info(f"Test config created at: {config_path}")
    else:
        config_path = args.config
        logger.info(f"Using provided config: {config_path}")
    
    # Load config
    cfg = load_config(config_path)
    
    # Test attribute access
    logger.info("Testing attribute access:")
    logger.info(f"cfg.circuit.qubits = {cfg.circuit.qubits}")
    logger.info(f"cfg.training.epochs = {cfg.training.epochs}")
    logger.info(f"cfg.data.protein_pairs = {cfg.data.protein_pairs}")
    
    # Test dictionary-style access
    logger.info("Testing dictionary-style access:")
    logger.info(f"cfg['circuit']['backend'] = {cfg['circuit']['backend']}")
    
    # Test get method
    logger.info("Testing get method:")
    logger.info(f"cfg.training.get('lr_decay') = {cfg.training.get('lr_decay')}")
    logger.info(f"cfg.training.get('nonexistent', 'default') = {cfg.training.get('nonexistent', 'default')}")
    
    # Test update
    logger.info("Testing update:")
    cfg.update({'new_section': {'param': 'value'}})
    logger.info(f"cfg.new_section.param = {cfg.new_section.param}")
    
    # Test to_dict conversion
    logger.info("Testing to_dict conversion:")
    config_dict = cfg.to_dict()
    logger.info(f"Type: {type(config_dict)}")
    logger.info(f"Keys: {list(config_dict.keys())}")
    
    # Test save (if temp file)
    if args.config is None:
        output_path = Path(temp_path).parent / 'saved_config.yaml'
        save_config(cfg, output_path)
        logger.info(f"Config saved to: {output_path}")
        
        # Reload and verify
        reloaded_cfg = load_config(output_path)
        logger.info(f"Reloaded cfg.circuit.qubits = {reloaded_cfg.circuit.qubits}")
    
    logger.info("Config loader test completed successfully")