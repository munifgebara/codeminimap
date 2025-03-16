import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';

// Caminho do modelo e da imagem
const MODEL_PATH = 'modelos_js/model.json';
const CLASS='other';
const IMAGE_DIR = '/home/munif-gebara-junior/ml/learn/codeminimap/dataset/novo_encrypted/'+CLASS;


async function carregarModelo() {
    try {
        // Carregar o modelo
        const model = await tf.loadGraphModel(`file://${MODEL_PATH}`);
        console.log("✅ Modelo carregado com sucesso!");

        // Ler todas as imagens da pasta
        const imageFiles = fs.readdirSync(IMAGE_DIR)
            .filter(file => file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg'));

        console.log(`📂 Encontradas ${imageFiles.length} imagens na pasta "${IMAGE_DIR}"`);

        for (const file of imageFiles) {
            const imagePath = path.join(IMAGE_DIR, file);
            await preverImagem(model, imagePath);
        }
    } catch (error) {
        console.error("❌ Erro ao carregar o modelo:", error);
    }
}

async function preverImagem(model, imagePath) {
    try {
        // Carregar e processar a imagem
        const imageBuffer = fs.readFileSync(imagePath);
        let imageTensor = tf.node.decodeImage(imageBuffer)
            .resizeNearestNeighbor([128, 128])  // Ajuste para o tamanho do modelo
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(255));  // Normaliza entre 0 e 1

        // Fazer a previsão
        const logits = model.predict(imageTensor);

        // Aplicar Softmax para converter logits em probabilidades
        const probabilities = tf.softmax(logits).dataSync();

        // Encontrar a classe com maior probabilidade
        const predictedClassIndex = probabilities.indexOf(Math.max(...probabilities));
        const confidence = Math.max(...probabilities) * 100; // Converter para %

        // Definir os nomes das classes (ajuste conforme necessário)
        const classNames = ['html', 'javaunittest', 'json', 'other', 'sql', 'svg', 'xml'];

        let classe = classNames[predictedClassIndex]
        if (classe != CLASS) {
            console.log(`📌 Imagem: ${path.basename(imagePath)}`);
            console.log(`🧐 Classe prevista: ${classe} (Índice: ${predictedClassIndex})`);
            console.log(`🔢 Confiança: ${confidence.toFixed(2)}%\n`);
        }
    } catch
        (error) {
        console.error(`❌ Erro ao processar a imagem "${imagePath}":`, error);
    }
}

// Executar o script
carregarModelo();