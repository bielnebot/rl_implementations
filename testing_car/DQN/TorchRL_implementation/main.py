
def train():
    pass


def test():
    pass


def main(args):
    if args.mode == "train":
        train()
    else:
        test()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", type=str, help="train or test")
    # args = parser.parse_args()

    # main(args)
    train()