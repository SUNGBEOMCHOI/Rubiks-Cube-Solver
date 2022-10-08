import yaml

def test(cfg):
    """
    
    """
    pass



if __name__ == "__main__":
    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    test(cfg)