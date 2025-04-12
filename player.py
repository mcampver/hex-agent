import time
import random
import heapq
from math import sqrt, log
from board import HexBoard

# Direcciones válidas en Hex
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1)]
class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def play(self, board) -> tuple:
        raise NotImplementedError("¡Implementa este método!")

class Node:
    def __init__(self, board, move, parent, player_id):
        self.board = board
        self.move = move
        self.parent = parent
        self.player_id = player_id
        self.children = []
        self.wins = 0
        self.visits = 0
        # Estadísticas AMAF/RAVE
        self.amaf_stats = {}
        # Valor posicional
        self.pos_value = None

    def uct_value(self, explore_param=1.4):
        if self.visits == 0:
            return float('inf')
        
        # RAVE stats
        amaf_wins, amaf_visits = (0, 0)
        if self.parent and self.move:
            amaf_wins, amaf_visits = self.parent.amaf_stats.get(self.move, (0, 0))
        
        # UCT clásico
        exploit = self.wins / self.visits
        
        # Factor de mezcla RAVE (más bajo con más visitas)
        k = 1000  # Parámetro de descuento
        beta = sqrt(k / (3 * self.visits + k))
        
        # Valor RAVE 
        amaf_val = amaf_wins / (amaf_visits + 1e-6) if amaf_visits > 0 else 0.5
        
        # Exploración UCT
        explore = explore_param * sqrt(log(self.parent.visits) / self.visits) if self.parent else 0
        
        # Mezcla ponderada UCT y RAVE
        return (1 - beta) * exploit + beta * amaf_val + explore

    def select(self):
        if not self.children:
            return self
        return max(self.children, key=lambda c: c.uct_value()).select()

    def expand(self):
        tried = {c.move for c in self.children}
        possible = [m for m in self.board.get_possible_moves() if m not in tried]
        
        if not possible:
            return self
        
        # Evaluación posicional de movimientos
        scored = []
        sz = self.board.size
        mid = sz // 2
        
        for move in possible:
            r, c = move
            # Distancia al centro
            dist = abs(r - mid) + abs(c - mid)
            # Vecinos amigos
            friends = 0
            # Vecinos enemigos
            enemies = 0
            
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < sz and 0 <= nc < sz:
                    cell = self.board.board[nr][nc]
                    if cell == self.player_id:
                        friends += 1
                    elif cell == 3 - self.player_id:
                        enemies += 1
            
            # Puntaje compuesto (centralidad + conectividad)
            score = (sz - dist) + 2*friends - enemies
            
            # Bonus direccional
            if self.player_id == 1:  # Horizontal
                score += c/2  # Avanzar hacia derecha
            else:  # Vertical
                score += r/2  # Avanzar hacia abajo
                
            scored.append((score, move))
        
        # Selección semi-aleatoria con preferencia a buenos movimientos
        scored.sort(reverse=True)
        if random.random() < 0.75 and len(scored) > 2:
            # Selecciona del top 30% con alta probabilidad
            idx = random.randint(0, max(0, len(scored)//3 - 1))
            move = scored[idx][1]
        else:
            move = random.choice(scored)[1]
        
        new_board = self.board.clone()
        new_board.place_piece(*move, self.player_id)
        child = Node(new_board, move, self, 3 - self.player_id)
        self.children.append(child)
        return child

    def simulate(self):
        sim_board = self.board.clone()
        player = self.player_id
        moves_played = []
        
        # Evitar simulaciones infinitas
        max_moves = sim_board.size * sim_board.size
        moves_count = 0
        
        # Simulación hasta ganar o alcanzar límite
        while not sim_board.check_connection(1) and not sim_board.check_connection(2):
            moves = sim_board.get_possible_moves()
            if not moves or moves_count >= max_moves:
                return 0.5, moves_played
                
            # Simulación semi-inteligente
            if random.random() < 0.7:  # Uso de heurística
                candidates = []
                best_val = -9999
                
                for m in moves:
                    r, c = m
                    val = 0
                    
                    # Conectividad
                    for dr, dc in DIRS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < sim_board.size and 0 <= nc < sim_board.size:
                            cell = sim_board.board[nr][nc] 
                            if cell == player:
                                val += 3  # Bonus por adyacencia a pieza propia
                            elif cell == 3 - player:
                                val += 1  # Pequeño bonus por bloquear
                    
                    # Dirección de avance
                    if player == 1:  # Horizontal
                        val += c  # Avance hacia derecha
                    else:  # Vertical
                        val += r  # Avance hacia abajo
                    
                    if val > best_val:
                        best_val = val
                        candidates = [m]
                    elif val == best_val:
                        candidates.append(m)
                
                move = random.choice(candidates)
            else:
                move = random.choice(moves)
            
            sim_board.place_piece(*move, player)
            moves_played.append((player, move))
            player = 3 - player
            moves_count += 1
        
        # Determina resultado para el jugador inicial
        if sim_board.check_connection(self.player_id):
            return 1.0, moves_played  # Victoria
        else:
            return 0.0, moves_played  # Derrota

    def backpropagate(self, result, moves_played):
        self.visits += 1
        self.wins += result
        
        # Actualiza AMAF stats en ancestros
        if self.parent:
            for (p, move) in moves_played:
                # Solo actualiza si el jugador coincide con el que jugaría en este nodo
                if p == self.player_id:
                    # Actualiza si el movimiento sigue siendo legal en el padre
                    if move in self.parent.board.get_possible_moves():
                        w, v = self.parent.amaf_stats.get(move, (0, 0))
                        self.parent.amaf_stats[move] = (w + result, v + 1)
            
            # Retropropaga invirtiendo el resultado
            self.parent.backpropagate(1 - result, moves_played)

    def best_child(self):
        if not self.children:
            return None
        
        # El hijo más visitado es más robusto
        return max(self.children, key=lambda c: c.visits)

class HexPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.tiempo_limite = 10  # Un poco menos para evitar timeouts
        self.oponente = 3 - player_id
        self.libro_aperturas = self._crear_libro_aperturas()
        self.patrones_observados = []  # Para seguimiento de jugadas del oponente
        
    def _crear_libro_aperturas(self):
        """Crea un pequeño libro de aperturas para las primeras jugadas"""
        libro = {}
        # Para tablero estándar (11x11)
        # Primera jugada: centro ligeramente desplazado
        libro[(11, 0)] = (5, 5)
        
        # Para tablero pequeño (7x7)
        libro[(7, 0)] = (3, 3)  
        
        return libro
        
    def _es_primera_jugada(self, tablero: HexBoard) -> bool:
        """Retorna True si el tablero está vacío"""
        return all(c == 0 for fila in tablero.board for c in fila)
    
    def _es_segunda_jugada(self, tablero: HexBoard) -> bool:
        """Retorna True si sólo hay una ficha en el tablero"""
        fichas = sum(1 for fila in tablero.board for c in fila if c != 0)
        return fichas == 1
        
    def _obtener_fichas_oponente(self, tablero: HexBoard) -> list:
        """Encuentra todas las posiciones de las fichas del oponente"""
        fichas = []
        for r in range(tablero.size):
            for c in range(tablero.size):
                if tablero.board[r][c] == self.oponente:
                    fichas.append((r, c))
        return fichas

    def _detectar_linea_horizontal(self, tablero: HexBoard, jugador: int) -> list:
        """Detecta si hay una línea horizontal formándose"""
        tam = tablero.size
        amenazas = []
        
        # Revisar cada fila
        for r in range(tam):
            segmentos = []  # Lista de segmentos de línea
            segmento_actual = []
            
            for c in range(tam):
                if tablero.board[r][c] == jugador:
                    segmento_actual.append((r, c))
                else:
                    if len(segmento_actual) >= 2:  # Consideramos amenaza desde 2 fichas contiguas
                        segmentos.append(segmento_actual)
                    segmento_actual = []
            
            # No olvidar el último segmento
            if len(segmento_actual) >= 2:
                segmentos.append(segmento_actual)
            
            # Analizar cada segmento encontrado
            for segmento in segmentos:
                # Comprobar si podemos extender el segmento
                col_min = min(c for _, c in segmento)
                col_max = max(c for _, c in segmento)
                
                # Mirar posición a la izquierda
                if col_min > 0 and tablero.board[r][col_min-1] == 0:
                    amenazas.append(((r, col_min-1), 50 + len(segmento)*10))
                
                # Mirar posición a la derecha
                if col_max < tam-1 and tablero.board[r][col_max+1] == 0:
                    amenazas.append(((r, col_max+1), 50 + len(segmento)*10))
        
        return amenazas
    
    def _detectar_linea_vertical(self, tablero: HexBoard, jugador: int) -> list:
        """Detecta si hay una línea vertical formándose"""
        tam = tablero.size
        amenazas = []
        
        # Revisar cada columna
        for c in range(tam):
            segmentos = []
            segmento_actual = []
            
            for r in range(tam):
                if tablero.board[r][c] == jugador:
                    segmento_actual.append((r, c))
                else:
                    if len(segmento_actual) >= 2:
                        segmentos.append(segmento_actual)
                    segmento_actual = []
            
            # No olvidar el último segmento
            if len(segmento_actual) >= 2:
                segmentos.append(segmento_actual)
            
            # Analizar cada segmento encontrado
            for segmento in segmentos:
                # Comprobar si podemos extender el segmento
                fila_min = min(r for r, _ in segmento)
                fila_max = max(r for r, _ in segmento)
                
                # Mirar posición arriba
                if fila_min > 0 and tablero.board[fila_min-1][c] == 0:
                    amenazas.append(((fila_min-1, c), 50 + len(segmento)*10))
                
                # Mirar posición abajo
                if fila_max < tam-1 and tablero.board[fila_max+1][c] == 0:
                    amenazas.append(((fila_max+1, c), 50 + len(segmento)*10))
        
        return amenazas
    
    def _detectar_patrones_oponente(self, tablero: HexBoard) -> list:
        """Analiza el historial de movimientos para identificar patrones"""
        fichas_oponente = self._obtener_fichas_oponente(tablero)
        
        # Actualizar historial de patrones observados
        self.patrones_observados = fichas_oponente
        
        # Verificar patrón de línea horizontal
        filas = {}
        for r, c in fichas_oponente:
            filas[r] = filas.get(r, 0) + 1
        
        # Columnas para patrón vertical
        columnas = {}
        for r, c in fichas_oponente:
            columnas[c] = columnas.get(c, 0) + 1
        
        amenazas = []
        
        # Identificar concentración en filas (patrón horizontal)
        for fila, cuenta in filas.items():
            if cuenta >= 3:  # 3 o más fichas en la misma fila
                # Encontrar huecos en esta fila para bloquear
                for c in range(tablero.size):
                    if tablero.board[fila][c] == 0:
                        # Calcular prioridad en base a continuidad
                        continuidad = 0
                        for dc in [-1, 1]:
                            nc = c + dc
                            if 0 <= nc < tablero.size and tablero.board[fila][nc] == self.oponente:
                                continuidad += 1
                        
                        prioridad = 80 + cuenta*10 + continuidad*20
                        amenazas.append(((fila, c), prioridad))
        
        # Identificar concentración en columnas (patrón vertical)
        for columna, cuenta in columnas.items():
            if cuenta >= 3:  # 3 o más fichas en la misma columna
                for r in range(tablero.size):
                    if tablero.board[r][columna] == 0:
                        # Calcular prioridad en base a continuidad
                        continuidad = 0
                        for dr in [-1, 1]:
                            nr = r + dr
                            if 0 <= nr < tablero.size and tablero.board[nr][columna] == self.oponente:
                                continuidad += 1
                        
                        prioridad = 80 + cuenta*10 + continuidad*20
                        amenazas.append(((r, columna), prioridad))
        
        return amenazas
        
    def busqueda_a_estrella(self, tablero, jugador):
        """Implementa A* para encontrar camino más corto entre bordes"""
        tam = tablero.size
        visitados = set()
        cola = []
        costos = {}
        
        def heuristica(fila, col):
            # Distancia Manhattan al borde opuesto
            return tam - 1 - (col if jugador == 1 else fila)
        
        def es_meta(fila, col):
            # Llegamos al lado opuesto?
            return (col == tam - 1) if jugador == 1 else (fila == tam - 1)
        
        # Añadimos las casillas del borde inicial
        for i in range(tam):
            fila, col = (i, 0) if jugador == 1 else (0, i)
            celda = tablero.board[fila][col]
            
            if celda == jugador:
                costo = 0  # Casilla propia: paso gratis
            elif celda == 0:
                costo = 1  # Casilla vacía: coste 1
            else:
                continue  # Casilla enemiga: no usamos
            
            heapq.heappush(cola, (costo + heuristica(fila, col), costo, fila, col))
            costos[(fila, col)] = costo
        
        # Búsqueda A*
        while cola:
            _, costo, fila, col = heapq.heappop(cola)
            if (fila, col) in visitados:
                continue
            
            visitados.add((fila, col))
            
            if es_meta(fila, col):
                return costo  # Camino encontrado
            
            for df, dc in DIRS:
                nf, nc = fila + df, col + dc
                if 0 <= nf < tam and 0 <= nc < tam and (nf, nc) not in visitados:
                    celda = tablero.board[nf][nc]
                    if celda == jugador:
                        nuevo_costo = costo  # Paso por casilla propia
                    elif celda == 0:
                        nuevo_costo = costo + 1  # Paso por casilla vacía
                    else:
                        nuevo_costo = costo + 1000  # Casilla enemiga (casi imposible)
                        
                    if (nf, nc) not in costos or nuevo_costo < costos[(nf, nc)]:
                        costos[(nf, nc)] = nuevo_costo
                        prioridad = nuevo_costo + heuristica(nf, nc)
                        heapq.heappush(cola, (prioridad, nuevo_costo, nf, nc))
        
        return float('inf')  # No hay camino
    
    def detectar_jugadas_criticas(self, tablero):
        """Detecta jugadas críticas ofensivas y defensivas"""
        jugadas_criticas = []
        
        # Detectar amenazas lineales
        amenazas_h = self._detectar_linea_horizontal(tablero, self.oponente)
        amenazas_v = self._detectar_linea_vertical(tablero, self.oponente)
        
        # Combinamos todas las amenazas
        todas_amenazas = amenazas_h + amenazas_v
        
        # Análisis de patrones del oponente
        amenazas_patron = self._detectar_patrones_oponente(tablero)
        todas_amenazas.extend(amenazas_patron)
        
        # Análisis tradicional con A*
        mi_costo = self.busqueda_a_estrella(tablero, self.player_id)
        costo_rival = self.busqueda_a_estrella(tablero, self.oponente)
        
        # ¿El oponente está cerca de ganar?
        if costo_rival <= 3:
            # Busca jugadas defensivas que bloqueen
            for jugada in tablero.get_possible_moves():
                tablero_temp = tablero.clone()
                tablero_temp.place_piece(*jugada, self.player_id)
                nuevo_costo_rival = self.busqueda_a_estrella(tablero_temp, self.oponente)
                if nuevo_costo_rival > costo_rival:
                    prioridad = 150 + (nuevo_costo_rival - costo_rival) * 20
                    jugadas_criticas.append((jugada, prioridad))
        
        # ¿Estamos cerca de ganar?
        if mi_costo <= 3:
            for jugada in tablero.get_possible_moves():
                tablero_temp = tablero.clone()
                tablero_temp.place_piece(*jugada, self.player_id)
                nuevo_mi_costo = self.busqueda_a_estrella(tablero_temp, self.player_id)
                if nuevo_mi_costo < mi_costo:
                    prioridad = 200 + (mi_costo - nuevo_mi_costo) * 30
                    jugadas_criticas.append((jugada, prioridad))
        
        # Añadir las amenazas lineales y de patrón a las jugadas críticas
        for (jugada, prioridad) in todas_amenazas:
            if jugada in tablero.get_possible_moves():
                jugadas_criticas.append((jugada, prioridad))
        
        return jugadas_criticas
    
    def encontrar_puentes(self, tablero):
        """Busca patrones de puente para identificar conexiones virtuales"""
        puentes = []
        tam = tablero.size
        
        # Patrones de puente básicos
        patrones = [
            [(0,1), (1,0)],  # Puente tipo "/"
            [(0,-1), (1,0)], # Puente tipo "\"
            [(0,1), (-1,0)], # Inversos
            [(0,-1), (-1,0)]
        ]
        
        for r in range(tam):
            for c in range(tam):
                if tablero.board[r][c] == self.player_id:
                    # Busca patrones de puente desde esta celda
                    for patron in patrones:
                        # Verifica las dos direcciones del patrón
                        celdas_puente = []
                        valido = True
                        
                        for dr, dc in patron:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < tam and 0 <= nc < tam:
                                if tablero.board[nr][nc] == 0:
                                    celdas_puente.append((nr, nc))
                                else:
                                    valido = False
                                    break
                            else:
                                valido = False
                                break
                        
                        # Verifica que la celda diagonal también sea propia
                        if valido and len(celdas_puente) == 2:
                            dr_total = patron[0][0] + patron[1][0]
                            dc_total = patron[0][1] + patron[1][1]
                            nr, nc = r + dr_total, c + dc_total
                            
                            if 0 <= nr < tam and 0 <= nc < tam and tablero.board[nr][nc] == self.player_id:
                                puentes.append(celdas_puente)
        
        return puentes
    
    def _respuesta_estrategica_segunda_jugada(self, tablero, ficha_oponente):
        """Respuesta más inteligente a la segunda jugada"""
        r, c = ficha_oponente
        tam = tablero.size
        
        # Si el oponente juega en el borde, bloquear directamente
        if r == 0 or r == tam-1 or c == 0 or c == tam-1:
            # Si juega en la primera fila, colocar ficha adyacente
            if r <= 1:  # Primera o segunda fila
                for dc in [-1, 0, 1]:
                    nc = c + dc
                    if 0 <= nc < tam and tablero.board[r][nc] == 0:
                        return (r, nc)  # Bloqueo horizontal
            
            # Si juega en la primera columna, colocar ficha adyacente
            if c <= 1:  # Primera o segunda columna
                for dr in [-1, 0, 1]:
                    nr = r + dr
                    if 0 <= nr < tam and tablero.board[nr][c] == 0:
                        return (nr, c)  # Bloqueo vertical
        
        # Si juega en otra posición, usar estrategia semi-espejo avanzada
        centro = tablero.size // 2
        
        # Si oponente juega cerca del centro, jugar adyacente
        dist_centro = abs(r - centro) + abs(c - centro)
        if dist_centro <= 2:
            # Buscar celda adyacente más conectada
            mejor = None
            mejor_val = -1
            
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < tam and 0 <= nc < tam and tablero.board[nr][nc] == 0:
                    # Evaluar valor posicional
                    val = 0
                    
                    # Proximidad al borde propio
                    if self.player_id == 1:  # Horizontal
                        val += (tam - 1 - abs(nr - centro))  # Centralidad en la fila
                    else:  # Vertical
                        val += (tam - 1 - abs(nc - centro))  # Centralidad en la columna
                    
                    if val > mejor_val:
                        mejor_val = val
                        mejor = (nr, nc)
            
            if mejor:
                return mejor
        
        # Si ninguna estrategia específica, semi-espejo modificado
        espejo_r = tam - 1 - r
        espejo_c = tam - 1 - c
        ajustado_r = (espejo_r + centro) // 2
        ajustado_c = (espejo_c + centro) // 2
        
        if tablero.board[ajustado_r][ajustado_c] == 0:
            return (ajustado_r, ajustado_c)
        
        # Fallback: centro del tablero
        return (centro, centro)
    
    def _evaluar_jugada_estrategica(self, tablero):
        """Busca jugada estratégica basada en la posición global"""
        tam = tablero.size
        centro = tam // 2
        mejor_jugada = None
        mejor_puntuacion = float('-inf')
        
        # Verificar si es mejor atacar o defender
        mi_costo = self.busqueda_a_estrella(tablero, self.player_id)
        costo_rival = self.busqueda_a_estrella(tablero, self.oponente)
        
        modo_ataque = mi_costo <= costo_rival
        
        for jugada in tablero.get_possible_moves():
            fila, col = jugada
            puntos = 0
            
            # Centralidad ajustada según dirección de juego
            if self.player_id == 1:  # Horizontal
                dist_centro_fila = abs(fila - centro)
                puntos += (tam - dist_centro_fila) * 2  # Centralidad en la fila es importante
                puntos += col  # Avance hacia la derecha
            else:  # Vertical
                dist_centro_col = abs(col - centro)
                puntos += (tam - dist_centro_col) * 2  # Centralidad en la columna
                puntos += fila  # Avance hacia abajo
            
            # Conectividad con piezas propias
            for df, dc in DIRS:
                nf, nc = fila + df, col + dc
                if 0 <= nf < tam and 0 <= nc < tam:
                    if tablero.board[nf][nc] == self.player_id:
                        puntos += 5  # Alta prioridad a conectar con piezas propias
            
            # Valor estratégico (simulación rápida)
            tablero_temp = tablero.clone()
            tablero_temp.place_piece(fila, col, self.player_id)
            
            if modo_ataque:
                # En modo ataque, valorar nuestro avance
                nuevo_costo = self.busqueda_a_estrella(tablero_temp, self.player_id)
                puntos += (mi_costo - nuevo_costo) * 10  # Bonus por reducir nuestro camino
            else:
                # En modo defensa, valorar bloquear al oponente
                nuevo_costo_rival = self.busqueda_a_estrella(tablero_temp, self.oponente)
                puntos += (nuevo_costo_rival - costo_rival) * 8  # Bonus por alargar camino rival
            
            if puntos > mejor_puntuacion:
                mejor_puntuacion = puntos
                mejor_jugada = jugada
        
        return mejor_jugada
                
    def play(self, tablero: HexBoard) -> tuple:
        # Verificar tiempo de inicio para evitar timeouts
        tiempo_inicio = time.time()
        
        # Jugada inicial según libro de aperturas
        if self._es_primera_jugada(tablero):
            if (tablero.size, 0) in self.libro_aperturas:
                return self.libro_aperturas[(tablero.size, 0)]
            # Sin apertura en libro: centro
            return (tablero.size // 2, tablero.size // 2)
            
        # Segunda jugada: respuesta estratégica mejorada
        if self._es_segunda_jugada(tablero):
            ficha_oponente = self._obtener_fichas_oponente(tablero)[0]
            return self._respuesta_estrategica_segunda_jugada(tablero, ficha_oponente)
        
        # Detectar jugadas críticas con máxima prioridad
        jugadas_criticas = self.detectar_jugadas_criticas(tablero)
        if jugadas_criticas:
            # Ordenar por prioridad y devolver la mejor jugada crítica
            jugadas_criticas.sort(key=lambda x: x[1], reverse=True)
            if jugadas_criticas[0][1] >= 100:  # Solo usar si es realmente crítico
                return jugadas_criticas[0][0]
        
        # Busca puentes (conexiones virtuales) para completar
        puentes = self.encontrar_puentes(tablero)
        if puentes and random.random() < 0.7:  # 70% de probabilidad para diversificar
            # Completa un puente aleatorio si hay alguno disponible
            for celdas_puente in puentes:
                for celda in celdas_puente:
                    if celda in tablero.get_possible_moves():
                        return celda
        
        # Tiempo restante disponible para MCTS
        # Tiempo restante disponible para MCTS
        tiempo_restante = self.tiempo_limite - (time.time() - tiempo_inicio)
        
        # Evaluar jugada estratégica si queda poco tiempo
        if tiempo_restante < 0.5:
            return self._evaluar_jugada_estrategica(tablero)
        
        # Si hay tiempo suficiente, usar MCTS con AMAF mejorado
        raiz = Node(tablero.clone(), None, None, self.player_id)
        iteraciones = 0
        
        # Inyectar conocimiento sobre jugadas críticas en el árbol MCTS
        if jugadas_criticas:
            # Inicializar valores AMAF para jugadas críticas
            for jugada, prioridad in jugadas_criticas:
                valor_normalizado = min(1.0, prioridad / 300.0)  # Normalizar a [0,1]
                raiz.amaf_stats[jugada] = (valor_normalizado * 10, 10)  # Simular visitas previas
        
        # Ejecuta MCTS hasta agotar tiempo
        while time.time() - tiempo_inicio < tiempo_restante:
            nodo = raiz.select()
            # Si no es estado terminal
            if not nodo.board.check_connection(1) and not nodo.board.check_connection(2):
                nodo = nodo.expand()
                resultado, jugadas = nodo.simulate()
                nodo.backpropagate(resultado, jugadas)
            iteraciones += 1
        
        mejor = raiz.best_child()
        
        # Fallback con heurística si MCTS no encuentra jugada
        if mejor is None:
            # Intentar primero una jugada estratégica
            jugada_estrategica = self._evaluar_jugada_estrategica(tablero)
            if jugada_estrategica:
                return jugada_estrategica
                
            # Última opción: heurística básica
            mejor_jugada = None
            mejor_puntuacion = float('-inf')
            
            for jugada in tablero.get_possible_moves():
                fila, col = jugada
                puntos = 0
                
                # Centralidad
                centro = tablero.size // 2
                dist_centro = abs(fila - centro) + abs(col - centro)
                puntos += (tablero.size - dist_centro)
                
                # Conectividad con piezas propias
                for df, dc in DIRS:
                    nf, nc = fila + df, col + dc
                    if 0 <= nf < tablero.size and 0 <= nc < tablero.size:
                        if tablero.board[nf][nc] == self.player_id:
                            puntos += 5
                        elif tablero.board[nf][nc] == self.oponente:
                            puntos += 1  # Pequeño bonus por adyacencia a enemigo
                
                # Avance direccional
                if self.player_id == 1:  # Horizontal
                    puntos += col * 2  # Bonus por avanzar a la derecha
                else:  # Vertical
                    puntos += fila * 2  # Bonus por avanzar hacia abajo
                
                if puntos > mejor_puntuacion:
                    mejor_puntuacion = puntos
                    mejor_jugada = jugada
            
            return mejor_jugada
            
        return mejor.move