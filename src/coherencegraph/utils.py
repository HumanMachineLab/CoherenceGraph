import torch

def check_similarity(emb1, emb2, coherence_threshold=0.5):
    # check similarity and add to coherent dictionary
    similarity = torch.cosine_similarity(
        emb1.reshape(1, -1), emb2.reshape(1, -1)
    )

    # print(f"Similarity: {similarity}")
    if similarity[0] >= coherence_threshold:
        return True
    return False
    
def get_similarity(emb1, emb2):
    # get similarity and add to coherent dictionary
    return torch.cosine_similarity(
        emb1.reshape(1, -1), emb2.reshape(1, -1)
    )