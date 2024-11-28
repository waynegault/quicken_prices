from box import Box

config = {"key": "value"}
box_config = Box(config)
print(box_config.key)  # Should output: value
