src="https://unpkg.com/brain.js"

        // 1. Criar a configuração da Rede Neural
        const net = new brain.NeuralNetwork();

        // 2. Treinar com os dados do XOR (que você já conhece)
        net.train([
            { input: [0, 0], output: [0] },
            { input: [0, 1], output: [1] },
            { input: [1, 0], output: [1] },
            { input: [1, 1], output: [0] }
        ]);

        // 3. Executar um teste
        const output = net.run([1, 0]); 

