from nvdya_kit.all_features import *

# Example: Streaming
streamer = StreamingProcessor(source='kafka')
streamer.process("sample stream data")

# Example: GNN
gnn = GNN(model_name='graph_sage')
gnn.train("sample graph data")
gnn.predict("sample graph data")

# Example: PPO
ppo = PPO(env_name='CartPole-v1')
ppo.train(episodes=10)
ppo.evaluate(episodes=2)

# Add more examples for each feature as needed... 