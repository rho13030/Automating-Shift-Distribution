import random
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Genetic Algorithm Functions
def generate_initial_population(num_workers, num_days, shifts_per_day, population_size):
    """Create an initial population of schedules."""
    population = []
    for _ in range(population_size):
        schedule = []
        for _ in range(num_days):
            day_schedule = [random.randint(0, num_workers - 1) for _ in range(shifts_per_day)]
            schedule.append(day_schedule)
        population.append(schedule)
    return population

def fitness(schedule, num_workers, max_shifts_per_day):
    """Calculate fitness of a schedule."""
    penalties = 0
    worker_counts = {worker: 0 for worker in range(num_workers)}

    for day_index, day in enumerate(schedule):
        daily_counts = {worker: 0 for worker in range(num_workers)}
        for shift_index, worker in enumerate(day):
            daily_counts[worker] += 1
            worker_counts[worker] += 1

            # Penalty for consecutive shifts
            if shift_index > 0 and day[shift_index] == day[shift_index - 1]:
                penalties += 50  # Big penalty for immediate consecutive shifts

            # Penalty for shifts close together (1-3 shifts apart)
            for gap in range(1, 4):
                if shift_index + gap < len(day) and day[shift_index] == day[shift_index + gap]:
                    penalties += 1000 * gap  # Penalty increases with smaller gaps

        # Penalty for exceeding max shifts per day
        penalties += sum(max(0, count - max_shifts_per_day) for count in daily_counts.values())

        # Penalty for late-night to early-morning shifts
        if day_index < len(schedule) - 1:  # Check next day's schedule
            if day[-1] in schedule[day_index + 1][:1]:
                penalties += 15  # Penalty for late-night to early-morning shifts

        # Penalty for last shift of a day to first shift of two days later
        if day_index < len(schedule) - 2:  # Check two days later
            if day[-1] == schedule[day_index + 2][0]:
                penalties += 200  # Big penalty for last shift -> first shift (2 days later)

    # Penalty for uneven shift distribution
    max_shifts = max(worker_counts.values())
    min_shifts = min(worker_counts.values())
    penalties += (max_shifts - min_shifts)

    return -penalties  # Lower penalties mean better fitness

def selection(population, fitness_scores):
    """Select parents based on fitness (roulette wheel selection)."""
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected = random.choices(population, probabilities, k=2)
    return selected

def crossover(parent1, parent2):
    """Perform single-point crossover between two parents."""
    point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(schedule, num_workers, mutation_rate):
    """Mutate a schedule by randomly changing some shifts."""
    for day in schedule:
        if random.random() < mutation_rate:
            shift = random.randint(0, len(day) - 1)
            day[shift] = random.randint(0, num_workers - 1)

            # Ensure no immediate consecutive shifts
            if shift > 0 and day[shift] == day[shift - 1]:
                day[shift] = (day[shift] + 1) % num_workers
            if shift < len(day) - 1 and day[shift] == day[shift + 1]:
                day[shift] = (day[shift] + 1) % num_workers
    return schedule

def genetic_algorithm(num_workers, num_days, shifts_per_day, max_shifts_per_day, generations=100, population_size=50, mutation_rate=0.1):
    """Run the genetic algorithm to optimize the schedule."""
    population = generate_initial_population(num_workers, num_days, shifts_per_day, population_size)

    for _ in range(generations):
        fitness_scores = [fitness(schedule, num_workers, max_shifts_per_day) for schedule in population]
        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, num_workers, mutation_rate)
            child2 = mutate(child2, num_workers, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    best_schedule = max(population, key=lambda s: fitness(s, num_workers, max_shifts_per_day))
    return best_schedule

# Tkinter GUI to input workers and generate schedule
class ShiftSchedulerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shift Scheduler")
        
        self.num_workers_label = tk.Label(root, text="Enter number of workers:")
        self.num_workers_label.pack()
        
        self.num_workers_entry = tk.Entry(root)
        self.num_workers_entry.pack()
        
        self.worker_names_label = tk.Label(root, text="Enter worker names (comma-separated):")
        self.worker_names_label.pack()
        
        self.worker_names_entry = tk.Entry(root)
        self.worker_names_entry.pack()
        
        self.generate_button = tk.Button(root, text="Generate Schedule", command=self.generate_schedule)
        self.generate_button.pack()

        self.tree = ttk.Treeview(root)
        self.tree.pack(expand=True, fill=tk.BOTH)

    def generate_schedule(self):
        try:
            # Get user input
            num_workers = int(self.num_workers_entry.get())
            worker_names = self.worker_names_entry.get().split(',')
            worker_names = [name.strip() for name in worker_names]

            if len(worker_names) != num_workers:
                raise ValueError("The number of worker names does not match the input number.")

            # Schedule settings
            num_days = 30
            shift_blocks = [
                "06:00-08:00", "08:00-10:00", "10:00-12:00", "12:00-14:00",
                "14:00-16:00", "16:00-18:00", "18:00-20:00", "20:00-22:00"
            ]
            shifts_per_day = len(shift_blocks)
            max_shifts_per_day = 2

            # Run Genetic Algorithm
            best_schedule = genetic_algorithm(
                num_workers, num_days, shifts_per_day, max_shifts_per_day
            )

            # Display Schedule
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = [f"Day {i+1}" for i in range(num_days)]
            self.tree.heading("#0", text="Shift Blocks", anchor="w")
            self.tree.column("#0", anchor="w")

            for i in range(num_days):
                self.tree.heading(f"Day {i+1}", text=f"Day {i+1}")
                self.tree.column(f"Day {i+1}", anchor="center")

            for shift_index, block in enumerate(shift_blocks):
                values = [worker_names[best_schedule[day_index][shift_index]] for day_index in range(num_days)]
                self.tree.insert("", "end", text=block, values=values)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")

# Run the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = ShiftSchedulerApp(root)
    root.mainloop()
