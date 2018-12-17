def to_surpervised(train, n_input=48, n_out=24):
    train_x , train_y = list(), list()
    in_start = 0
    for _ in range(len(train)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end < len(train):
            train_x.append(train[in_start: in_end, :])
            train_y.append(train[in_end: out_end, 0])
        in_start += n_out
    return train_x, train_y
