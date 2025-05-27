from nvdya_kit.all_features import *

def test_streaming():
    s = StreamingProcessor()
    assert "processed_data" in s.process("test")

def test_gnn():
    g = GNN()
    assert "status" in g.train("test")
    assert "prediction" in g.predict("test")

def test_ppo():
    p = PPO()
    assert "status" in p.train(1)
    assert "reward" in p.evaluate(1)

# Add more tests for each feature... 