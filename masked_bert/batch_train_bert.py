from train_masked_bert import train

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    for p in [.4,.8,0,.2,.6,.1,.05,.025]:
        train(checkpoint_path=args.checkpoint_path, masked_p=p, version_id=f'v7_p{p}')
