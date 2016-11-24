import mnist_loader
import network
import cPickle
import gzip
import shutil


def save_network(_):
    output = open('../data/net.pkl', 'w')

    # Pickle dictionary using protocol 0.
    cPickle.dump(_, output)

    output.close()

    with open('../data/net.pkl', 'r') as f_in, gzip.open('../data/net.pkl.gz', 'w') as f_out:
        shutil.copyfileobj(f_in, f_out)


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

save_network(net.export_net())