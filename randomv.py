import random
import tkinter as tk
from tkinter import ttk

def generate_monthly_schedule(shift_blocks, workers, max_shifts_per_day=4, avoid_consecutive=True, days_in_month=30):
    # Initialize the schedule for the month
    schedule = {day: {block: None for block in shift_blocks} for day in range(days_in_month)}
    worker_shifts = {worker: [0]*days_in_month for worker in workers}  # Track daily shifts per worker
    total_shifts = {worker: 0 for worker in workers}  # Track total shifts for fairness

    for day in range(days_in_month):
        for block in shift_blocks:
            # Get workers available for the current block
            available_workers = [
                worker for worker in workers
                if worker_shifts[worker][day] < max_shifts_per_day and
                   (not avoid_consecutive or 
                    (day == 0 or block not in schedule[day-1].values())) and
                   total_shifts[worker] < (days_in_month * len(shift_blocks)) // len(workers)
            ]
            # Assign a worker to the shift
            if available_workers:
                assigned_worker = random.choice(available_workers)
                schedule[day][block] = assigned_worker
                worker_shifts[assigned_worker][day] += 1
                total_shifts[assigned_worker] += 1

    return schedule

# Input Parameters
shift_blocks = [
    "06:00-08:00", "08:00-10:00", "10:00-12:00", "12:00-14:00",
    "14:00-16:00", "16:00-18:00", "18:00-20:00", "20:00-22:00",
    "04:00-06:00"
]
workers = ["이충호", "나대현", "오재현", "노우진", "오재현"]
max_shifts_per_day = 4
days_in_month = 30

# Generate the schedule
monthly_schedule = generate_monthly_schedule(shift_blocks, workers, max_shifts_per_day, days_in_month=days_in_month)

# Create the GUI
root = tk.Tk()
root.title("Monthly Shift Schedule")

# Create a treeview to display the schedule
tree = ttk.Treeview(root)
tree["columns"] = ("Day", *shift_blocks)
tree.heading("#0", text="", anchor="w")
tree.column("#0", anchor="w", width=0)
for block in shift_blocks:
    tree.heading(block, text=block)
    tree.column(block, anchor="center")

# Insert data into the treeview
for day, shifts in monthly_schedule.items():
    values = [day + 1] + [shifts[block] for block in shift_blocks]
    tree.insert("", "end", text="", values=values)

# Pack the treeview
tree.pack(expand=True, fill=tk.BOTH)

# Run the GUI
root.mainloop()
