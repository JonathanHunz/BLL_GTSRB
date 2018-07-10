from src.training import FeedForward

ffw = FeedForward("gtsrb_3", batch_size=128, learning_rate=0.0005, iterations=624, train_capacity=2500, test_capacity=2500)

ffw.run()

