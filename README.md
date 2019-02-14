# HTUNEML - machine learning experiments monitoring and tuning

**Quickstart:** ``pip install htuneml``. See the "Installing" section for more details.


Project links:

-  `PyPI <https://pypi.python.org/pypi/htuneml/>`__
-  `Source code <https://github.com/johnsmithm/htuneml>`__
-  `Issue tracker <https://github.com/johnsmithm/htuneml/issues>`__


Examples
--------

See the ``examples/`` `directory <https://github.com/htuneml/htuneml/tree/master/examples>`__ in the repository root for usage examples:

-  `Tensorflow Mnist <https://github.com/htuneml/htuneml/blob/master/examples/tensorflow-mnist.py>`__
-  `Keras Mnist <https://github.com/htuneml/htuneml/blob/master/examples/tensorflow-mnist.py>`__
-  `Pytorch Mnist <https://github.com/htuneml/htuneml/blob/master/examples/tensorflow-mnist.py>`__


Requirements
------------

To use all of the functionality of the library, you should have:

* **Python** 2.6, 2.7, or 3.3+ (required)
* **PyAudio** 0.2.11+ (required only if you need to use microphone input, ``Microphone``)


Quick start
------------

Register on website http://registru.ml, copy the api_key:

.. code:: python

    import htuneml as ht
    
    job = Job('api_key')
    
    @job.monitor
    def train(par1=2,par2=2):
        for i in range(par1):
            #do training here
            job.log({'loss':i*4,'ep':i})

    job.setName('l2')
    #job.debug()# uncomment and no experiment will be created and no logs sent
    train(10, 2)

This will print out something like the following:

::

    make experiment
    got key experimnet 5c5c8eaacbcfb9146641367a

Also it is possible to sent the parameters from the web app. First on gpu/cpu set the lisener:

.. code:: python

    import htuneml as ht
    
    job = Job('api_key')
    
    def train(par1=2,par2=2):
        for i in range(par1):
            #do training here
            job.log({'loss':i*4,'ep':i})

    job.sentParams(train)#sent the parameters list to the app
    job.waitTask(train)#wait for parameters from app
    
