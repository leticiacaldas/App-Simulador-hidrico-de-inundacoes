/**
* Nome do modelo: Simulacao de Inundacao 3D - Versão Simplificada
* Autor: Leticia & Copilot
* Descrição: Modelo que usa um arquivo GeoTIFF para criar uma grade de agentes.
*            A lógica de inicialização foi removida para máxima compatibilidade.
* GAMA version: 1.9.2
*/
model SimulacaoInundacao3D

global {
    // Arquivo GeoTIFF com o modelo de elevação digital (DEM)
    file dem_file <- grid_file("../data/DOM_AOI.tif", false);

    // Geometria que define o contorno do mundo
    geometry shape <- envelope(dem_file);

    // Parâmetros da simulação
    float rain_amount <- 0.01; // Quantidade de chuva por passo de tempo
    
    // Valores fixos para a cor do terreno. Ajuste se necessário.
    // Você pode descobrir os valores min/max olhando a legenda do seu arquivo TIF em um software de GIS (como o QGIS).
    float min_elev_para_cor <- 220.0;
    float max_elev_para_cor <- 280.0;

    // O bloco 'init' é intencionalmente deixado vazio.
    // A criação e inicialização dos agentes é feita automaticamente pela declaração da 'species'.
    init {}

    // Reflexo global para simular a chuva
    reflex chover {
        ask celula_terreno {
            lamina_dagua <- lamina_dagua + rain_amount;
        }
    }
}

// A espécie 'celula_terreno' É a grade, carregada a partir do arquivo.
// Cada célula da grade se torna um agente 'celula_terreno'.
species celula_terreno skills: [grid] {
    
    // O único atributo dinâmico que precisamos armazenar.
    float lamina_dagua <- 0.0;

    // Aspecto para visualização 3D
    aspect default {
        // 'grid_value' é um atributo automático que contém o valor da célula da grade (nossa elevação).
        float minha_elevacao <- grid_value;
        
        // Calcula a cor do terreno em tempo real.
        rgb minha_cor_terreno;
        if (max_elev_para_cor > min_elev_para_cor) {
            float normalized_elev <- (minha_elevacao - min_elev_para_cor) / (max_elev_para_cor - min_elev_para_cor);
            minha_cor_terreno <- hsb(0.2 - (normalized_elev * 0.2), 0.7, 0.8);
        } else {
            minha_cor_terreno <- #green;
        }
        
        // Desenha a base do terreno na sua elevação.
        draw square(1) color: minha_cor_terreno at: {location.x, location.y, minha_elevacao};
        
        // Se houver água, desenha uma camada azul por cima.
        if (lamina_dagua > 0.001) {
            draw square(1) color: rgb(100, 150, 255, 0.7) at: {location.x, location.y, minha_elevacao + lamina_dagua};
        }
    }
}

// Experimento com visualização 3D
experiment main_3d type: gui {
    output {
        display main_display type: 3d {
            // Apenas diz ao GAMA para desenhar os agentes.
            // O 'aspect' de cada agente cuida de toda a lógica de desenho.
            species celula_terreno aspect: default;
        }
    }
}