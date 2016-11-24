import cPickle
import gzip
import network
import mnist_loader


def load_network():
    # type: () -> network
    f = gzip.open('../data/net.pkl.gz', 'rb')
    num_layers, sizes, biases, weights = cPickle.load(f)
    f.close()
    # Init network
    net = network.Network([])
    # Set network
    net.import_net(num_layers, sizes, biases, weights)

    return net

def test_network_loader(net):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print "{0} / {1}".format(
        net.evaluate(test_data), len(test_data))

# Test the network import
# test_network_loader(load_network())