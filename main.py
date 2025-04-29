import time
import random
from multiprocessing import Process, Queue
import math
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import ListProperty, BooleanProperty, VariableListProperty, ColorProperty, OptionProperty, \
    NumericProperty, StringProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivymd.theming import ThemableBehavior
from kivymd.tools.hotreload.app import MDApp
from kivymd.uix import MDAdaptiveWidget
from kivymd.uix.behaviors import DeclarativeBehavior, BackgroundColorBehavior, CommonElevationBehavior, \
    RectangularRippleBehavior
from kivymd.uix.behaviors.state_layer_behavior import StateLayerBehavior
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.progressindicator import MDLinearProgressIndicator
from kivymd.uix.screen import MDScreen

# Generate 100 random inputs between 1 and 1000
inputs = [random.randint(1, 1000) for _ in range(5000)]


def processor(input_value: float | int):
    """
    Processor function that performs CPU-intensive calculations:
    1. Fibonacci sequence calculation
    2. Prime number detection
    3. Matrix operations with deterministic values
    4. Complex mathematical computations
    """
    def calculate_fibonacci(n):
        if n <= 1:
            return n
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def matrix_multiply(size):
        # Create two deterministic matrices based on input_value
        matrix1 = [[((i + j) % 10) + 1 for j in range(size)] for i in range(size)]
        matrix2 = [[(i * j % 10) + 1 for j in range(size)] for i in range(size)]
        
        # Multiply matrices
        result = [[sum(a * b for a, b in zip(row, col)) 
                  for col in zip(*matrix2)] 
                  for row in matrix1]
        return result

    # Perform multiple intensive calculations
    result = 0
    
    # 1. Calculate Fibonacci sequence
    fib_n = min(input_value % 20, 35)  # Limit to avoid excessive recursion
    fib_result = calculate_fibonacci(fib_n)
    
    # 2. Find prime numbers up to input_value (limited to prevent excessive computation)
    max_prime_check = min(input_value, 1000)
    prime_count = sum(1 for num in range(2, max_prime_check) if is_prime(num))
    
    # 3. Perform matrix multiplication with deterministic values
    matrix_size = min(int(math.sqrt(input_value)), 20)  # Limit matrix size
    matrix_result = matrix_multiply(matrix_size)
    matrix_sum = sum(sum(row) for row in matrix_result)
    
    # 4. Perform additional mathematical operations
    complex_math = (math.cos(input_value) * 1000 + 
                   math.tan(input_value % 1.5) * 100 +
                   math.exp(input_value % 7))
    
    # Combine all results
    result = (fib_result + 
             prime_count + 
             matrix_sum + 
             complex_math)
    
    return result


class App(MDApp):
    DEBUG = True
    AUTORELOADER_PATHS = [
        (".", {"recursive": False}),
        ("./KV", {"recursive": True}),
    ]
    KV_DIRS = [
        "./KV"
    ]

    def build_app(self, first=False):
        return MainScreen()


class MainScreen(MDScreen):
    def start_serial_processing(self):
        """Execute serial processing on all inputs"""
        import threading
        from queue import Queue

        # Get references to UI elements
        progress_bar = self.ids.serial_progress_bar
        time_indicator = self.ids.serial_time_indicator
        serial_start_button = self.ids.serial_processing_button
        parallel_start_button = self.ids.parallel_processing_button

        # Disable buttons while processing
        serial_start_button.disabled = True
        parallel_start_button.disabled = True

        # Reset progress values
        progress_bar.values = [0]
        time_indicator.time_elapsed = 0
        time_indicator.processing_time = 0

        # Create a queue for communication between the processing thread and UI updater
        progress_queue = Queue()
        processing_done = [False]  # Use a list for mutable reference
        processing_end_time = [0]  # To store when processing is complete

        # Start the timer for tracking elapsed time
        start_time = time.time()

        # This function will update the UI and runs in the main thread
        def update_ui(dt):
            # Update elapsed time
            current_time = time.time()
            elapsed = current_time - start_time
            time_indicator.time_elapsed = elapsed

            # Check if we need to update the progress percentage
            if not progress_queue.empty():
                progress_percentage = progress_queue.get()
                progress_bar.values = [progress_percentage]

            # Check if processing is complete
            if processing_done[0]:
                # Processing complete
                end_time = processing_end_time[0]
                time_indicator.processing_time = end_time - start_time

                # Re-enable buttons
                serial_start_button.disabled = False
                parallel_start_button.disabled = False

                return False  # Stop the clock schedule

            return True  # Continue the clock schedule

        # Define the worker function that will run in a separate thread
        def processing_worker():
            processing_start = time.time()
            for i, input_val in enumerate(inputs):
                # Process this input
                result = processor(input_val)

                # Update progress (0-100 for KivyMD)
                progress_percentage = ((i + 1) / len(inputs)) * 100
                print(f"Progress: {progress_percentage}%")
                progress_queue.put(progress_percentage)

            # Mark processing as complete
            processing_done[0] = True
            processing_end_time[0] = time.time()

        # Start the processing in a separate thread
        processing_thread = threading.Thread(target=processing_worker)
        processing_thread.daemon = True  # Thread will exit when main program exits
        processing_thread.start()

        # Schedule the UI update function to run periodically
        Clock.schedule_interval(update_ui, 0.05)  # Update UI 20 times per second

    def start_processing(self, progress_bar, time_indicator, num_processes):
        """Execute parallel processing on all inputs using multiprocessing"""
        from multiprocessing import Manager, Process
        from kivy.clock import mainthread  # Import mainthread decorator

        # Disable buttons while processing
        self.ids.serial_processing_button.disabled = True
        self.ids.parallel_processing_button.disabled = True

        # Initialize progress values
        progress_bar.values = [0] * num_processes
        time_indicator.time_elapsed = 0
        time_indicator.processing_time = 0

        # Create manager as a class attribute to keep it alive
        self._manager = Manager()
        
        # Create shared lists for progress and completion status
        shared_progress = self._manager.list([0] * num_processes)
        shared_completed = self._manager.list([False] * num_processes)

        # Prepare the work distribution
        chunk_size = len(inputs) // num_processes
        chunks = []

        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_processes - 1 else len(inputs)
            chunks.append((start_idx, end_idx))

        # Define worker function for each process
        def worker_process(process_id, chunk_indices, shared_progress, shared_completed):
            start_idx, end_idx = chunk_indices
            total_items = end_idx - start_idx

            try:
                for i, idx in enumerate(range(start_idx, end_idx)):
                    result = processor(inputs[idx])
                    # Update shared progress directly (0-1 range)
                    progress = (i + 1) / total_items
                    shared_progress[process_id] = progress

                print(f"{process_id} Done processing chunk {start_idx} to {end_idx}")
            except Exception as e:
                print(f"Error in process {process_id}: {e}")
            finally:
                # Mark this process as complete when done
                shared_completed[process_id] = True

        # Start processes
        processes = []
        start_time = time.time()

        for i in range(num_processes):
            p = Process(
                target=worker_process,
                args=(i, chunks[i], shared_progress, shared_completed)
            )
            processes.append(p)
            p.start()

        # Define a mainthread function to update the progress bar
        @mainthread
        def update_progress_bar(values):
            print(f"Progress bar values: {progress_bar.values}")
            progress_bar.values = values

        def update_progress(dt):
            # Update elapsed time
            current_time = time.time()
            elapsed = current_time - start_time
            time_indicator.time_elapsed = elapsed

            try:
                # Update UI progress bars from shared memory
                progress_values = []
                for i in range(num_processes):
                    # Get current progress from shared memory
                    progress = shared_progress[i]

                    # If process completed, ensure progress shows 100%
                    if shared_completed[i] and progress < 0.99:
                        progress = 1.0

                    progress_values.append(progress * 100)  # Convert to 0-100 scale for UI

                # Update progress bar using the mainthread decorator
                update_progress_bar(progress_values)

                # Debug print to verify progress values
                print(f"Progress values: {progress_values}")
                progress_bar.values = progress_values

                # If all processes are complete
                if all(shared_completed):
                    # Processing complete
                    end_time = time.time()
                    time_indicator.processing_time = end_time - start_time

                    # Clean up processes
                    for p in processes:
                        if p.is_alive():
                            p.join(0.1)

                    # Re-enable buttons
                    self.ids.serial_processing_button.disabled = False
                    self.ids.parallel_processing_button.disabled = False

                    # Clean up manager
                    self._manager.shutdown()
                    del self._manager

                    return False  # Stop the clock schedule

            except (AttributeError, EOFError) as e:
                print(f"Error updating progress: {e}")  # Add debug print
                return False

            return True  # Continue the clock schedule

        # Schedule the update function to run periodically
        Clock.schedule_interval(update_progress, 0.05)  # Update 20 times/second


class SingleProgressBar(MDBoxLayout):
    indicator_color = ColorProperty([1, 0, 1, 1])
    value = NumericProperty(0.0)


class ProcessingProgressBar(MDBoxLayout):
    num_processes = NumericProperty(9)
    values = ListProperty()
    indicator_color = ColorProperty([1, 0, 1, 1])
    indicator_bg_color = ColorProperty([1, 0, 0, 1])

    def on_num_processes(self, *args):

        while len(self.values) != self.num_processes:
            if len(self.values) > self.num_processes:
                self.values.pop(0)
            elif len(self.values) < self.num_processes:
                self.values.append(0)


    def on_values(self, *args):
        # Ensure the correct number of progress bars are present.
        while len(self.children) != len(self.values):
            # print(len(self.values), self.num_processes)
            if len(self.children) < len(self.values):
                self.add_widget(SingleProgressBar(
                ))
            elif len(self.children) > len(self.values):
                self.remove_widget(self.children[-1])

        for i, progress_bar in enumerate(self.children):
            progress_bar.value = self.values[i]


class TimeIndicator(MDBoxLayout):
    time_elapsed = NumericProperty(0.0)
    processing_time = NumericProperty(0.0)


class PerformanceResult(MDBoxLayout):
    metric_name = StringProperty()
    metric_value = NumericProperty()
    metric_unit = StringProperty()
    icon = StringProperty()
    icon_color = ColorProperty([1, 0, 1, 1])
    metric_description = StringProperty()

if __name__ == "__main__":
    App().run()