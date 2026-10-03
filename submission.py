from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, board_size, delivery_reward_multiplier
import random
import time
from func_timeout import func_timeout, FunctionTimedOut
TIME_MARGIN = 0.1
# ANSI escape codes for text colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

# ANSI codes for text styles
BOLD = "\033[1m"
UNDERLINE = "\033[4m"


EXPECTIMAX_ACTION_WEIGHTS = {
    "move north": 4,
    "charge": 4,
}

TIE_BREAKING_ORDER = [
    "drop off",
    "pick up",
    "charge",
    "move north",
    "move east",
    "move south",
    "move west",
    "park",
]
def op_weight(op):
    return EXPECTIMAX_ACTION_WEIGHTS.get(op,1)

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    package = robot.package
    battery = robot.battery

    # חישוב הבסיס: הפרש הנקודות בינינו לבין היריב.
    # זהו הרכיב הכי חשוב באלגוריתמי סכום-אפס כמו מינימקס ואלפא-בטא!
    base_score = 200*(robot.credit - other_robot.credit)

    cs_positions = [cs.position for cs in env.charge_stations]

    # battery incentive
    battery_penalty = 0
    dead_rival = (other_robot.battery==0)
    block_condition=[]
    for p in env.packages[:2]:
        if p.destination == other_robot.position:
            block_condition.append(p)
    game_blocked = (block_condition==env.packages[:2] and other_robot.battery==0)
    if battery < 2*board_size and not game_blocked:
        battery_penalty-=(4000+1000*battery) 

    def dist_to_closest_cs(pos):
        return min([manhattan_distance(pos, cs_pos) for cs_pos in cs_positions])

    # מקרה 1: הרובוט מחזיק חבילה
    if package is not None:
        dist_to_dest = manhattan_distance(robot.position, package.destination)
        delivery_distance = manhattan_distance(package.position, package.destination)
        reward = delivery_reward_multiplier * delivery_distance
        dist_dest_to_ccs = dist_to_closest_cs(package.destination)

        # האם יש מספיק סוללה כדי להגיע ליעד ומשם לתחנת טעינה?
        if dist_to_dest + dist_dest_to_ccs < battery:
            state_val = 2000 + (100 * reward) - 100*dist_to_dest
            if (package.destination==other_robot.position and other_robot.battery==0):
                state_val+=float('-inf')
        else:
            # אין מספיק סוללה לסיים את המשימה. ננסה להינצל ע"י הגעה לתחנת טעינה,
            # אך ורק אם יש לנו קרדיט להמיר לסוללה!
            if robot.credit > 0:
                state_val = -100 * dist_to_closest_cs(robot.position)
            else:
                # אין סוללה ואין קרדיט - מצב אנוש
                state_val = float('-inf')

        return base_score + state_val + battery_penalty

    # מקרה 2: הרובוט לא מחזיק חבילה, מחפש את החבילה המשתלמת ביותר
    else:
        package_heuristics = []
        for p in env.packages[:2]:
            dist_to_pkg = manhattan_distance(robot.position, p.position)
            dist_pkg_to_dest = manhattan_distance(p.position, p.destination)
            dist_dest_to_cs = dist_to_closest_cs(p.destination)

            # חישוב מסלול מלא: הליכה לחבילה + הגעה ליעד + הגעה לתחנת טעינה לביטחון
            total_cost = dist_to_pkg + dist_pkg_to_dest + dist_dest_to_cs

            if total_cost < battery:
                if (other_robot.position in [p.position,p.destination] and other_robot.battery==0):
                    continue

                reward = delivery_reward_multiplier * dist_pkg_to_dest
                # הוספת הציון של החבילה לאפשרויות שלנו
                package_heuristics.append(1000 + (100 * reward) - 100*dist_to_pkg)

        # בדיקה האם יש חבילות שניתן לאסוף בבטחה (מונע את שגיאת ה-ValueError!)
        if len(package_heuristics) > 0:
            state_val = max(package_heuristics)
        else:
            # אין אף חבילה שאפשר לאסוף ולמסור בבטחה.
            # game over - no more credit can be gained dont waste any on charging
            # if (env.packages[0].position==env.packages[1].position==other_robot.position and other_robot.battery==0):
            #     return 
            # נלך להטעין אם יש לנו קרדיט:
            if robot.credit > 0:
                state_val = 500 - 100 * dist_to_closest_cs(robot.position)
            else:
                # אם אין קרדיט, סתם נלך לכיוון החבילה הקרובה ביותר
                closest_pkg_dist = min([manhattan_distance(robot.position, p.position) for p in env.packages[:2]])
                state_val = -100 * closest_pkg_dist

        return base_score + state_val +battery_penalty


# TODO: section b : fixed-depth helper for deterministic grading
def minimax_rec(env: WarehouseEnv, cur_turn: int, orig_robot: int, depth: int, heuristic_fn=None):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    operators = env.get_legal_operators(cur_turn)

    if env.done() or depth == 0 or not operators:
        return heuristic_fn(env, orig_robot), None

    next_turn = (cur_turn + 1) % 2

    if cur_turn == orig_robot:
        cur_max = float('-inf')
        max_op = None
        for op in TIE_BREAKING_ORDER:
            if op not in operators:
                continue
            next_state = env.clone()
            next_state.apply_operator(cur_turn, op)
            next_val, _ = minimax_rec(next_state, next_turn, orig_robot, depth - 1, heuristic_fn)
            if next_val > cur_max:
                cur_max = next_val
                max_op = op
        return cur_max, max_op

    else:
        cur_min = float('inf')
        min_op = None
        for op in TIE_BREAKING_ORDER:
            if op not in operators:
                continue
            next_state = env.clone()
            next_state.apply_operator(cur_turn, op)
            next_val, _ = minimax_rec(next_state, next_turn, orig_robot, depth - 1, heuristic_fn)
            if next_val < cur_min:
                cur_min = next_val
                min_op = op
        return cur_min, min_op

def minimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited minimax.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    opt_val, opt_op = minimax_rec(env, robot_id, robot_id, depth, heuristic_fn)
    return opt_op



# TODO: section c : fixed-depth helper for deterministic grading
def alphabeta_rec(env: WarehouseEnv, cur_turn: int, orig_robot: int, depth: int, alpha: float, beta: float, heuristic_fn=None):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    operators = env.get_legal_operators(cur_turn)

    if env.done() or depth == 0 or not operators:
        return heuristic_fn(env, orig_robot), None

    next_turn = (cur_turn + 1) % 2

    if cur_turn == orig_robot:
        cur_max = float('-inf')
        max_op = None
        for op in TIE_BREAKING_ORDER:
            if op not in operators:
                continue
            next_state = env.clone()
            next_state.apply_operator(cur_turn, op)
            next_val, _ = alphabeta_rec(next_state, next_turn, orig_robot, depth - 1, alpha, beta, heuristic_fn)
            if next_val > cur_max:
                cur_max = next_val
                max_op = op

            alpha = max(alpha, cur_max)
            if cur_max >= beta:
                break
        return cur_max, max_op

    else:
        cur_min = float('inf')
        min_op = None
        for op in TIE_BREAKING_ORDER:
            if op not in operators:
                continue
            next_state = env.clone()
            next_state.apply_operator(cur_turn, op)
            next_val, _ = alphabeta_rec(next_state, next_turn, orig_robot, depth - 1, alpha, beta, heuristic_fn)
            if next_val < cur_min:
                cur_min = next_val
                min_op = op

            beta = min(beta, cur_min)
            if cur_min <= beta:
                break

        return cur_min, min_op

def alphabeta_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited alpha-beta pruning.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    opt_val, opt_op = alphabeta_rec(env, robot_id, robot_id, depth, float('-inf'), float('inf'), heuristic_fn)
    return opt_op

# TODO: section d : fixed-depth helper for deterministic grading
def expectimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited expectimax.
    The opponent's legal actions are weighted by EXPECTIMAX_ACTION_WEIGHTS;
    every legal action not in the dictionary has weight 1.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    if heuristic_fn == None:
        heuristic_fn = smart_heuristic
    rival_id = 1-robot_id
    def expectimax_helper(curr_env: WarehouseEnv, curr_robot_id: int, curr_depth: int):
        if curr_env.done() or curr_depth<=0: return heuristic_fn(curr_env,robot_id)
        operators = curr_env.get_legal_operators(curr_robot_id)
        # opponent turn - chance node
        if curr_robot_id == rival_id:
            value = 0
            for op in operators:
                succ = curr_env.clone()
                succ.apply_operator(curr_robot_id,op)
                value += op_weight(op)*expectimax_helper(succ,robot_id,curr_depth-1)
                # value +=(op_weight(op)*h_val)
            
            return value/sum([op_weight(op) for op in operators])
        # our turn - max node
        else:
            max_value = float('-inf')
            best_op=None
            for op in operators:
                succ = curr_env.clone()
                succ.apply_operator(curr_robot_id,op)
                value = expectimax_helper(succ,rival_id,curr_depth-1)
                if value > max_value: best_op = op
                max_value = max(max_value,value)
            return max_value

    legal_ops = env.get_legal_operators(robot_id)
    max_value = float('-inf')
    best_op = legal_ops[0]
    for op in legal_ops:
        succ = env.clone()
        succ.apply_operator(robot_id,op)
        value = expectimax_helper(succ,rival_id,depth-1)
        if value > max_value:
            max_value = value
            best_op = op
        elif value == max_value and best_op!=None:
            if TIE_BREAKING_ORDER.index(op)<TIE_BREAKING_ORDER.index(best_op):
                best_op = op
    return best_op


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)
    

class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        best_operator = None
        current_depth = 1

        time_margins = time_limit * 0.1

        while (time.time() - start_time) < time_margins:
            op = minimax_decision(env, agent_id, current_depth)
            if op is not None:
                best_operator = op
            current_depth += 1

        # Fallback if best_operator was none
        if best_operator is None:
            operators = env.get_legal_operators(agent_id)
            for op in TIE_BREAKING_ORDER:
                if op in operators:
                    return op
            best_operator = "park"

        return best_operator

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        best_operator = None
        current_depth = 1

        time_margins = time_limit * 0.1

        while (time.time() - start_time) < time_margins:
            op = alphabeta_decision(env, agent_id, current_depth)
            if op is not None:
                best_operator = op
            current_depth += 1

        # Fallback if best_operator was none
        if best_operator is None:
            operators = env.get_legal_operators(agent_id)
            for op in TIE_BREAKING_ORDER:
                if op in operators:
                    return op
            best_operator = "park"

        return best_operator



class AgentExpectimax(Agent):
    # TODO: section d : 3
    # def run_step(self, env: WarehouseEnv, agent_id, time_limit):
    #     end_time = time.time()+time_limit
    #     epsilon = time_limit*0.8
    #     depth = 1
    #     op = None
    #     while True:
    #         if time.time()>=end_time-epsilon:
    #                 break   
    #                 # raise Exception("exceded time limit")
    #         try:
    #             op = expectimax_decision(env, agent_id,depth)
    #             depth+=1
    #         except TimeoutError:
    #             break
    #     return op
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        best_operator = None
        current_depth = 1
        safe_time = time_limit - TIME_MARGIN
        start_time = time.time()

        try:
            while True:
                # Calculate exactly how much time is left right NOW
                time_left = safe_time - (time.time() - start_time)
                if time_left <= 0:
                    break
                op = func_timeout(time_left, expectimax_decision, args=(env, agent_id, current_depth))
                # If we got here, func_timeout succeeded
                if op is not None:
                    best_operator = op
                current_depth += 1

        except FunctionTimedOut:
            pass

        # Fallback if best_operator was none
        if best_operator is None:
            operators = env.get_legal_operators(agent_id)
            for op in TIE_BREAKING_ORDER:
                if op in operators:
                    return op
            best_operator = "park"

        return best_operator

# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
