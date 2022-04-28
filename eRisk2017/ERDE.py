import numpy as np

def ERDE_chunk(chunk_pred_probas, chunk_cum_posts, labels, threshold=0.5, o=5):
    """
    chunk_pred_probas: predicted depression prob at each chunk, num_users x chunk_num
    chunk_cum_posts: used number of posts at each chunk, num_users x chunk_num
    labels: num_users
    """
    num_users, chunk_num = chunk_pred_probas.shape
    num_pos = np.sum(labels)
    erde = 0
    # since the result for each chunk is pre computed, we can put user outside
    for i in range(num_users):
        depress = labels[i]
        for chunk in range(chunk_num):
            early_stop = chunk_pred_probas[i][chunk] >= threshold
            # TP
            if depress and early_stop:
                erde+=(1-1.0/(1+np.exp(chunk_cum_posts[i][chunk]-o)))
            # FP
            elif not depress and early_stop:
                erde += num_pos/num_users
            # FN
            elif chunk == chunk_num-1 and depress and not early_stop:
                erde += 1
            if early_stop:
                break
    return erde / num_users

def ERDE_sample(sample_pred_probas, labels, threshold=0.5, o=5):
    """
    calculate ERDE at per sample level (predict at each sample)
    sample_pred_probas: predicted depression prob at each sample, num_users lists, each contains num_samples predictions 
    labels: num_users
    """
    num_users = len(labels)
    num_pos = np.sum(labels)
    erde_pre_user = []
    for user, pred_probas in enumerate(sample_pred_probas):
        erde0 = 0
        early_stop = False
        depress = labels[user]
        for k, proba in enumerate(pred_probas):
            early_stop = proba >= threshold
            # TP
            if depress and early_stop:
                erde0 = (1-1.0/(1+np.exp(k+1-o)))
            # FP
            elif not depress and early_stop:
                erde0 = num_pos/num_users
            if early_stop:
                break
        # FN
        if depress and not early_stop:
            erde0 = 1
        # print(k, erde0)
        erde_pre_user.append(erde0)
    return np.mean(erde_pre_user)

if __name__ == "__main__":
    sample_pred_probas = [
        [0.1, 0.8, 0.3, 0.4, 0.49, 0.53, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.49, 0.47, 0.42, 0.52],
        [0.1, 0.2, 0.3, 0.4, 0.49, 0.47, 0.42, 0.22, 0.8],
        [0.1, 0.2, 0.3, 0.1, 0.49, 0.47, 0.42, 0.32],
        [0.1, 0.2, 0.3, 0.4, 0.49, 0.47, 0.42, 0.2, 0.1, 0.4, 0.9],
    ]
    labels = [1,1,0,0,1]
    print(ERDE_sample(sample_pred_probas, labels))