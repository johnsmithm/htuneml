import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
import argparse
from htuneml import Job

mnist = tf.keras.datasets.mnist

def getA():
    parser = argparse.ArgumentParser(description='Test a network')
    parser.add_argument('-n', '--name', type=str, help='name experiment', default='test')
    parser.add_argument('-e', '--epochs', type=int, help='number epochs', default=1)
    parser.add_argument('-d', '--debug', type=int, help='if 1 debug/do not sent experiment', default=0)
    parser.add_argument('-w', '--wait', type=int, help='if 1 wait for task from web app', default=0)
    parser.add_argument('-l', '--hidden', type=int, help='number neurons hidden layer', default=512)
    args = parser.parse_args()    
    return args

job = Job('apikey')

@job.monitor
def main(pars=None):

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(pars['hidden'], activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    logs = LambdaCallback(on_epoch_end=lambda epoch, logs: job.log({'ep':epoch, 'acc':logs['acc'],'loss':logs['loss']}))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=pars['epochs'],callbacks=[logs])
    results = model.evaluate(x_test, y_test)
    
    print('loss, acc test:', results)
    job.log({'acc':results[1],'loss':results[0],'type':1})

if __name__ == '__main__':
    args = getA()
    pars = vars(args)
    print(pars)
    
    if pars['debug'] == 1:
        job.debug()#no logs will be sent to app
    else:
        job.setName(pars['name'])
        
    if pars['wait'] == 1:
        job.sentParams(main)# comment @job.monitor above main function in order to work
        job.waitTask(main)
    else:
        main(pars)
        