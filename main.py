import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import tensorflow as tf
from sklearn.neural_network import MLPRegressor



def normalize_data(m):
    (row_num, col_num) = m.shape

    # fit missing data
    result = np.nan_to_num(m)

    # do normalization
    # make each value within [-1, 1]
    # (although it not that exact in
    # the current implementation)
    for i in range(col_num):
        col = m[:, i]
        mean_val = col.mean()
        max_val = col.max()
        max_diff = max_val - mean_val if max_val - mean_val != 0 else 1

        for j in range(row_num):
            result[j][i] = 1.0 * (result[j][i] - mean_val) / max_diff

    return result



def neural_network_train(training_x, training_y, test_x, test_y):
    training_y = np.reshape(training_y, (training_y.shape[0],))
    test_y = np.reshape(test_y, (test_y.shape[0]))

    reg = MLPRegressor(algorithm='l-bfgs',
                       alpha=1e-5,
                       hidden_layer_sizes=(35, 35),
                       random_state=1,
                       activation="tanh",
                       max_iter=500)
    reg.fit(training_x, training_y)

    pred_y_test = reg.predict(test_x)
    pred_y_train = reg.predict(training_x)

    rs_test = cal_r_square(test_y, pred_y_test)
    rs_train = cal_r_square(training_y, pred_y_train)

    training_loss = get_square_error(training_y, pred_y_train)
    test_loss = get_square_error(test_y, pred_y_test)

    return {"training_r2": rs_train, "test_r2": rs_test, "training_loss": training_loss, "test_loss": test_loss}



def get_square_error(a, b):
    err = a - b
    err = err * err
    return err.sum()



def tensor_flow_train(training_x, training_y, test_x, test_y):
    # set up tensor flow
    feature_num = training_x.shape[1]
    x = tf.placeholder(tf.float32, [None, feature_num])
    y_ = tf.placeholder(tf.float32, [None, 1])  # y_ holds the real, observed output

    w = tf.Variable(tf.zeros([feature_num, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, w) + b

    diff = tf.squared_difference(y, y_)
    loss = tf.reduce_mean(diff)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    # run it
    sess = tf.Session()
    sess.run(init)

    steps = 6001
    for step in range(steps):
        sess.run(train, feed_dict={x: training_x, y_: training_y})

    # predict outputs
    predict_y_training = sess.run(y, feed_dict={x: training_x})
    predict_y_test = sess.run(y, feed_dict={x: test_x})

    # calculate R2
    rs_train = cal_r_square(training_y, predict_y_training)
    rs_test = cal_r_square(test_y, predict_y_test)

    # square loss
    test_loss = sess.run(loss, feed_dict={x: test_x, y_: test_y})
    training_loss = sess.run(loss, feed_dict={x: training_x, y_: training_y})

    result = {'training_r2': rs_train, 'test_r2': rs_test, 'test_loss': test_loss, 'training_loss': training_loss}

    return result



def prepare_data():
    """
    Randomly shuffle the data set and split it into training and test sets.
    :return: training and test data sets
    """
    # pre-process data
    raw = pd.read_csv("p1pm.csv", )
    raw = raw.drop(raw.columns[[0]], axis=1)
    # remove constant columns
    # raw = raw.drop(raw.columns[[12, 13, 14, 15]], axis=1)

    data = raw.as_matrix()
    np.random.shuffle(data)
    x_data = data[:, 1:]
    y_data = data[:, 0:1]

    x_data = np.nan_to_num(x_data).astype(np.float32)
    y_data = y_data.reshape([35, 1]).astype(np.float32)
    x_data = normalize_data(x_data)

    train_set_num = 30
    training_x = x_data[0: train_set_num]
    training_y = y_data[0: train_set_num]
    test_x = x_data[train_set_num:]
    test_y = y_data[train_set_num:]

    return {'training_x': training_x,
            'training_y': training_y,
            'test_x': test_x,
            'test_y': test_y}



def cal_r_square(real, pred):
    real_mean = np.mean(real)
    res = np.sum(np.square(real - pred))
    tot = np.sum(np.square(real - real_mean))

    return 1.0 - (res / tot) if tot != 0 else 0.0



def main():
    best = 99999
    best_result = None

    compare_key = 'test_loss'
    train_func = neural_network_train

    training_losses = np.array([])
    test_losses = np.array([])

    # try several different splits and pick the best
    n = 100
    for i in range(n):

        data = prepare_data()
        training_x = data['training_x']
        training_y = data['training_y']
        test_x = data['test_x']
        test_y = data['test_y']
        result = train_func(training_x, training_y, test_x, test_y)

        # store losses for scatter plot
        training_losses = np.append(training_losses, result['training_loss'])
        test_losses = np.append(test_losses, result['test_loss'])

        if result[compare_key] < best:
            best = result[compare_key]
            best_result = result
        if i % 10 == 0:
            print "best ", i, " = ", best
            print "training r2 = ", best_result['training_r2']
            print "test r2 = ", best_result['test_r2']
            print "training loss = ", best_result['training_loss']
            print "test loss = ", best_result['test_loss']
            print

    print 'Overall Best:'
    print "training r2 = ", best_result['training_r2']
    print "test r2 = ", best_result['test_r2']
    print "training loss = ", best_result['training_loss']
    print "test loss = ", best_result['test_loss']

    # plot losses
    plot.scatter(training_losses, test_losses)
    plot.show()


if __name__ == "__main__":
    main()
