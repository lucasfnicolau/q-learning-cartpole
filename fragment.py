import gym
import numpy as np

# Criação da q-table utilizando um dicionário
q_table = {}

# Hiper-parâmetros
GAMMA = 1 # Taxa de desconto
epsilon = 1 # Taxa exploração
alpha = 1 # Taxa de aprendizado

env = gym.make('CartPole-v1')

# Discretização e outras configurações
# ...

# Função de treinamento
def train(GAMMA, alpha, epsilon, iterations=3001):
    print('\nComeçando treino...')

    successful_episodes = 0

    for i in range(iterations):
        # Reinicializando o ambiente para
        # pegar a primeira observation
        observation = env.reset()

        # Pegando o estado equivalente a essa observação,
        # já com os dados discretizados
        state = get_state(assign_bins(observation, bins))

        done = False

        episode_length = 0

        while not done:
            # Definição da ação a ser realizada
            # utilizando a técnica greedy-epsilon
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # 'Explore'
            else:
                action = get_action(q_table[state]) # 'Exploit'

            episode_length += 1

            # Execução da ação escolhida no ambiente,
            # recebendo de volta uma nova observation,
            # uma recompensa, se o estado é terminal
            # e uma informação para fins de depuração
            observation, reward, done, _ = env.step(action)

            # Definição do novo estado, baseado na
            # observation discretizada
            next_state = get_state(assign_bins(observation, bins))

            # Verifica se o estado era terminal e caso
            # não vença o jogo (episódio com 500 movimentos):
            # define a recompensa como sendo -300,
            # para pesar negativamente, evitando que o
            # algoritmo repita essa escolha futuramente
            if done:
                if episode_length < 500:
                    reward = -300
                else:
                    successful_episodes += 1

            # Armazena o valor atual, já o considerando como o antigo
            old_value = q_table[state][action]

            # Busca o novo valor (q-value) máximo para o estado futuro
            next_max = get_max_q(q_table[next_state])

            # Calcula o novo valor utilizando a fórmula do Q-Learning
            new_value = alpha * (reward + GAMMA * next_max - old_value)

            # Atualiza a q-table adicionando o novo valor calculado
            # à posição [estado][action] atual
            q_table[state][action] += new_value

            # Atualiza o estado atual como sendo o novo
            state = next_state

            # Atualização dos hiper-parâmetros, diminuindo seus
            # valores a cada iteração, priorizando cada vez
            # mais os valores aprendidos
            epsilon = 1.0 / np.sqrt(i + 1)
            alpha = 1.0 / np.sqrt(i + 1)

        # Impressão na tela a cada 1000 iterações
        # informando quantos episódios foram um sucesso
        if i % 1000 == 0 and i > 0:
            clear_output(wait=True)
            print(f'Episódio: {i} | Número de sucessos: {successful_episodes}/1000')
            successful_episodes = 0

    print('Treino finalizado.\n')
