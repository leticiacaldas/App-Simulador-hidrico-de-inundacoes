# design.py - Template HTML para o Simulador Híbrido de Inundações

def get_html_template():
    return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulador Híbrido de Inundações</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #1e5799 0%, #207cca 100%);
            color: white;
            padding: 25px 30px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        h2 {
            font-size: 20px;
            margin-bottom: 15px;
            color: #1e5799;
            border-bottom: 1px solid #eaeaea;
            padding-bottom: 8px;
        }
        
        h3 {
            font-size: 18px;
            margin: 15px 0 10px;
            color: #207cca;
        }
        
        .section {
            padding: 20px 30px;
            border-bottom: 1px solid #eaeaea;
        }
        
        .section:last-child {
            border-bottom: none;
        }
        
        .simulation-type {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .checkbox-group {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 10px;
        }
        
        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        input[type="checkbox"] {
            width: 18px;
            height: 18px;
        }
        
        hr {
            border: none;
            border-top: 1px dashed #ccc;
            margin: 20px 0;
        }
        
        .input-data {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        
        .input-data ul {
            list-style-type: none;
            padding-left: 10px;
        }
        
        .input-data li {
            margin-bottom: 8px;
            padding-left: 15px;
            position: relative;
        }
        
        .input-data li:before {
            content: "•";
            position: absolute;
            left: 0;
            color: #207cca;
        }
        
        .parameters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .parameter-table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .parameter-table th {
            background-color: #eef5ff;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            color: #1e5799;
            border-bottom: 1px solid #ddd;
        }
        
        .parameter-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .parameter-table tr:last-child td {
            border-bottom: none;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .metric-table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .metric-table th {
            background-color: #eef5ff;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            color: #1e5799;
            border-bottom: 1px solid #ddd;
        }
        
        .metric-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .metric-table tr:last-child td {
            border-bottom: none;
        }
        
        .highlight {
            background-color: #fff9e6;
            font-weight: 600;
        }
        
        .control-panel {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background-color: #1e5799;
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #16457a;
        }
        
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background-color: #545b62;
        }
        
        @media (max-width: 768px) {
            .parameters, .metrics {
                grid-template-columns: 1fr;
            }
            
            .section {
                padding: 15px 20px;
            }
            
            .control-panel {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Simulador Híbrido de Inundações</h1>
            <div class="simulation-type">
                <h2>Simulação Vectorizada (NumPy)</h2>
            </div>
            <div class="checkbox-group">
                <div class="checkbox-item">
                    <input type="checkbox" id="elemento" checked>
                    <label for="elemento">Elemento (DIA)</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="texto1">
                    <label for="texto1">Texto</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="texto2">
                    <label for="texto2">Texto</label>
                </div>
            </div>
        </header>
        
        <hr>
        
        <div class="section">
            <h2>Dados de Entrada</h2>
            
            <h3>Aquinas e Reguladas</h3>
            <div class="input-data">
                <p><strong>Noutra Região de Emenda: 2014</strong></p>
                <ul>
                    <li>Doça para doze Dia Anos</li>
                    <li>Lunes 2020 por dia no TCP 7/8</li>
                    <li>BRAÇAIS_XXXIAT_1.2.100</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>Parâmetros da Simulação</h2>
            
            <div class="parameters">
                <table class="parameter-table">
                    <tr>
                        <th>Prestupidez</th>
                        <th>Comportamento da Água</th>
                    </tr>
                    <tr>
                        <td>O mesmo estilo para</td>
                        <td>Varela alusada</td>
                    </tr>
                    <tr>
                        <td>ESCAP</td>
                        <td>0,50</td>
                    </tr>
                </table>
                
                <table class="parameter-table">
                    <tr>
                        <th>Testa de dados</th>
                        <th>Listas de inovação (Tempo)</th>
                    </tr>
                    <tr>
                        <td>00</td>
                        <td>0,50</td>
                    </tr>
                </table>
                
                <table class="parameter-table">
                    <tr>
                        <th>Oração de usina este (Invanto)</th>
                        <th>Nota de chave</th>
                    </tr>
                    <tr>
                        <td>30</td>
                        <td>Utilizamos os anos</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2>Inovação na grada</h2>
            
            <div class="checkbox-group">
                <div class="checkbox-item">
                    <input type="checkbox" id="inovacao1">
                    <label for="inovacao1">Pós-investirado a água boba sobre animação</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="inovacao2">
                    <label for="inovacao2">Segundo o inovação</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="inovacao3">
                    <label for="inovacao3">Para o inovação</label>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Métricas em Tempo Real</h2>
            
            <div class="metrics">
                <table class="metric-table">
                    <tr>
                        <th>Tempo em Trabalho</th>
                        <th>Tempo de Inovação</th>
                        <th>Aterrorizada</th>
                        <th>Vazionar dúvida</th>
                    </tr>
                    <tr>
                        <td id="tempoTrabalho">N/A</td>
                        <td id="tempoInovacao1">0h 0m</td>
                        <td id="aterrorizada1">0,00%</td>
                        <td id="vazionar1">0,00 m³</td>
                    </tr>
                </table>
                
                <table class="metric-table">
                    <tr>
                        <th>Tempo de Inovação</th>
                        <th>Aterrorizada</th>
                        <th>Vazionar dúvida</th>
                    </tr>
                    <tr class="highlight">
                        <td id="tempoInovacao2">Dia 20m</td>
                        <td id="aterrorizada2">0,00%</td>
                        <td id="vazionar2">20,02 AES m³</td>
                    </tr>
                </table>
            </div>
            
            <div class="control-panel">
                <button class="btn btn-primary" onclick="iniciarSimulacao()">Iniciar Simulação</button>
                <button class="btn btn-secondary" onclick="pausarSimulacao()">Pausar</button>
                <button class="btn btn-secondary" onclick="resetarSimulacao()">Resetar</button>
                <button class="btn btn-primary" onclick="exportarDados()">Exportar Dados</button>
            </div>
        </div>
    </div>
    
    <script>
        // Variáveis de estado da simulação
        let simulacaoAtiva = false;
        let tempoDecorrido = 0;
        let intervaloSimulacao;
        
        // Função para atualizar métricas em tempo real
        function atualizarMetricas() {
            if (!simulacaoAtiva) return;
            
            tempoDecorrido += 1;
            
            // Atualizar tempos
            document.getElementById('tempoTrabalho').textContent = formatarTempo(tempoDecorrido);
            document.getElementById('tempoInovacao1').textContent = formatarTempo(tempoDecorrido * 0.5);
            document.getElementById('tempoInovacao2').textContent = `Dia ${Math.floor(tempoDecorrido / 24)}h ${tempoDecorrido % 24}m`;
            
            // Simular dados de inundação (valores aleatórios para demonstração)
            const aterrorizada1 = (Math.random() * 100).toFixed(2);
            const aterrorizada2 = (Math.random() * 50).toFixed(2);
            const vazionar1 = (Math.random() * 100).toFixed(2);
            const vazionar2 = (20 + Math.random() * 5).toFixed(2);
            
            document.getElementById('aterrorizada1').textContent = aterrorizada1 + '%';
            document.getElementById('aterrorizada2').textContent = aterrorizada2 + '%';
            document.getElementById('vazionar1').textContent = vazionar1 + ' m³';
            document.getElementById('vazionar2').textContent = vazionar2 + ' AES m³';
        }
        
        // Função para formatar tempo
        function formatarTempo(minutos) {
            const horas = Math.floor(minutos / 60);
            const mins = minutos % 60;
            return `${horas}h ${mins}m`;
        }
        
        // Funções de controle da simulação
        function iniciarSimulacao() {
            if (!simulacaoAtiva) {
                simulacaoAtiva = true;
                intervaloSimulacao = setInterval(atualizarMetricas, 1000);
                console.log('Simulação iniciada');
            }
        }
        
        function pausarSimulacao() {
            simulacaoAtiva = false;
            if (intervaloSimulacao) {
                clearInterval(intervaloSimulacao);
            }
            console.log('Simulação pausada');
        }
        
        function resetarSimulacao() {
            pausarSimulacao();
            tempoDecorrido = 0;
            
            // Resetar métricas
            document.getElementById('tempoTrabalho').textContent = 'N/A';
            document.getElementById('tempoInovacao1').textContent = '0h 0m';
            document.getElementById('tempoInovacao2').textContent = 'Dia 20m';
            document.getElementById('aterrorizada1').textContent = '0,00%';
            document.getElementById('aterrorizada2').textContent = '0,00%';
            document.getElementById('vazionar1').textContent = '0,00 m³';
            document.getElementById('vazionar2').textContent = '20,02 AES m³';
            
            console.log('Simulação resetada');
        }
        
        function exportarDados() {
            const dados = {
                tempoTrabalho: document.getElementById('tempoTrabalho').textContent,
                tempoInovacao1: document.getElementById('tempoInovacao1').textContent,
                tempoInovacao2: document.getElementById('tempoInovacao2').textContent,
                aterrorizada1: document.getElementById('aterrorizada1').textContent,
                aterrorizada2: document.getElementById('aterrorizada2').textContent,
                vazionar1: document.getElementById('vazionar1').textContent,
                vazionar2: document.getElementById('vazionar2').textContent
            };
            
            console.log('Dados exportados:', dados);
            alert('Dados exportados! Verifique o console para visualizar.');
            
            // Aqui você pode implementar o download real dos dados
            // const blob = new Blob([JSON.stringify(dados, null, 2)], {type: 'application/json'});
            // const url = URL.createObjectURL(blob);
            // const a = document.createElement('a');
            // a.href = url;
            // a.download = 'dados_simulacao.json';
            // a.click();
        }
        
        // Inicialização
        console.log('Simulador Híbrido de Inundações carregado');
        console.log('Use as funções: iniciarSimulacao(), pausarSimulacao(), resetarSimulacao(), exportarDados()');
    </script>
</body>
</html>
"""

def save_html_file(filename="simulador_inundacoes.html"):
    """Salva o template HTML em um arquivo"""
    html_content = get_html_template()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Arquivo {filename} salvo com sucesso!")

# Exemplo de uso
if __name__ == "__main__":
    save_html_file()