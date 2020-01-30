import {Cifar10} from './data.js';

async function load () {
    const data = new Cifar10()
    await data.load()

    const model = await tf.loadLayersModel('./model/model.json');
    //tfvis.show.modelSummary({name: 'Model Architecture'}, model);

    const class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    const [test_images, test_labels] = data.nextTestBatch();

    const idx = 0
    const imageTensor = tf.tidy(() => {
        return test_images.slice([idx, 0], [1, test_images.shape[1]]).reshape([32, 32, 3]);
    });

    const surface = tfvis.visor().surface({ name: 'Test image'});
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);
    imageTensor.dispose();

    const predictions = model.predict(test_images.slice(idx, idx + 1).reshape([1, 32, 32, 3]))
    const argmax = predictions.argMax(1).dataSync()[0];
    console.log(argmax);

    const predictionsArray = await predictions.array();

    for(let i=0; i<predictionsArray[0].length; i++) {
        const p = document.createElement('p');
        if(i == argmax)
            p.style = 'color: green;';
        else
            p.style = 'color: blue;';
        p.innerHTML = class_names[i] + ': ' + predictionsArray[0][i];
        surface.drawArea.appendChild(p);
    }
}

load();