import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
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

def encode_schedule(schedule, num_workers):
    """Encode the schedule as a one-hot representation."""
    encoded = []
    for day in schedule:
        encoded_day = []
        for shift in day:
            one_hot = [0] * num_workers
            one_hot[shift] = 1
            encoded_day.extend(one_hot)
        encoded.append(encoded_day)
    return np.array(encoded)

def decode_schedule(encoded_schedule, num_workers, shifts_per_day):
    """Decode the one-hot representation back to the schedule."""
    decoded_schedule = []
    for encoded_day in encoded_schedule:
        day = []
        for i in range(shifts_per_day):
            one_hot = encoded_day[i * num_workers:(i + 1) * num_workers]
            if len(one_hot) == 0:
                continue
            shift = np.argmax(one_hot)
            day.append(shift)
        decoded_schedule.append(day)
    return decoded_schedule

def prepare_data(population, num_workers, shifts_per_day):
    """Prepare the input and output data for the neural network."""
    X = []
    y = []
    for schedule in population:
        encoded_schedule = encode_schedule(schedule, num_workers)
        X.append(encoded_schedule[:-1])  # Previous days as input
        y.append(encoded_schedule[1:])   # Next day as output
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], shifts_per_day * num_workers))
    y = y.reshape((y.shape[0], y.shape[1], shifts_per_day * num_workers))
    return X, y

def create_model(input_shape, output_shape):
    """Create and compile the neural network model."""
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='softmax'))
    model.add(Reshape(output_shape))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def valid_assignment(schedule, day, shift, worker, num_workers, shifts_per_day):
    """Check if the assignment is valid with no consecutive shifts and a 3-shift gap."""
    # Check for consecutive shifts
    if shift > 0 and schedule[day][shift - 1] == worker:
        return False
    if shift < shifts_per_day - 1 and schedule[day][shift + 1] == worker:
        return False

    # Check for a 3-shift gap within the same day
    for i in range(1, 4):
        if shift - i >= 0 and schedule[day][shift - i] == worker:
            return False
        if shift + i < shifts_per_day and schedule[day][shift + i] == worker:
            return False

    # Check for a 3-shift gap on the previous day
    if day > 0:
        for i in range(1, 4):
            if shifts_per_day - i >= 0 and schedule[day - 1][shifts_per_day - i] == worker:
                return False

    return True

def generate_schedule_with_constraints(num_workers, num_days, shifts_per_day):
    """Generate a schedule with the specified constraints."""
    schedule = [[-1] * shifts_per_day for _ in range(num_days)]
    
    for day in range(num_days):
        for shift in range(shifts_per_day):
            available_workers = [worker for worker in range(num_workers) if valid_assignment(schedule, day, shift, worker, num_workers, shifts_per_day)]
            if available_workers:
                schedule[day][shift] = random.choice(available_workers)
    
    return schedule

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
            population_size = 50
            generations = 100

            # Generate initial population
            population = generate_initial_population(num_workers, num_days, shifts_per_day, population_size)
            X, y = prepare_data(population, num_workers, shifts_per_day)

            # Create and train the model
            model = create_model(input_shape=(X.shape[1], X.shape[2]), output_shape=(y.shape[1], y.shape[2]))
            model.fit(X, y, epochs=generations, verbose=1)

            # Predict the schedule
            encoded_best_schedule = model.predict(X[0:1])[0]
            best_schedule = decode_schedule(encoded_best_schedule, num_workers, shifts_per_day)

            # Ensure best_schedule has the correct dimensions
            while len(best_schedule) < num_days:
                best_schedule.append([0] * shifts_per_day)  # Padding with default values
            if len(best_schedule) > num_days:
                best_schedule = best_schedule[:num_days]
            
            for day in best_schedule:
                while len(day) < shifts_per_day:
                    day.append(0)  # Padding with default values
                if len(day) > shifts_per_day:
                    day = day[:shifts_per_day]

            # Modify the schedule to ensure no consecutive shifts and a 3-shift gap
            for day in range(num_days):
                for shift in range(shifts_per_day):
                    worker = best_schedule[day][shift]
                    if not valid_assignment(best_schedule, day, shift, worker, num_workers, shifts_per_day):
                        available_workers = [w for w in range(num_workers) if valid_assignment(best_schedule, day, shift, w, num_workers, shifts_per_day)]
                        if available_workers:
                            best_schedule[day][shift] = random.choice(available_workers)

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
        except IndexError as e:
            messagebox.showerror("Index Error", f"Index out of range: {e}")
            print(f"Error in best_schedule dimensions: {best_schedule}")

# Run the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = ShiftSchedulerApp(root)
    root.mainloop()
