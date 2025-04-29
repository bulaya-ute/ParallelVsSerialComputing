import time
import random
from multiprocessing import Process, Queue
import math
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import ListProperty, BooleanProperty, VariableListProperty, ColorProperty, OptionProperty, \
    NumericProperty
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
inputs = [random.randint(1, 1000) for _ in range(50)]


def processor(input_value: float | int):
    """
    Processor function that simulates complex calculation by:
    1. Computing multiple mathematical operations
    2. Introducing a small delay to simulate processing time
    """
    # Simulate complex processing with a delay proportional to input value
    delay = (input_value % 10) / 10  # 0.0 to 0.9 seconds
    time.sleep(delay)

    # Perform some calculations
    result = math.sqrt(input_value) + math.sin(input_value) * 10 + math.log(input_value + 1)

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

    def start_parallel_processing(self):
        """Execute parallel processing on all inputs using multiprocessing"""
        from multiprocessing import Manager, Process

        # Get references to UI elements
        progress_bar = self.ids.parallel_progress_bar
        time_indicator = self.ids.parallel_processing_button.parent.children[0]

        # Get number of processes
        try:
            num_processes = int(self.ids.num_processes_textfield.text)
            if num_processes <= 0:
                num_processes = 2
        except (ValueError, TypeError):
            num_processes = 2
            self.ids.num_processes_textfield.text = "2"

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

        # Monitor process progress and update UI
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

                # Update progress bar all at once to avoid visual artifacts
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

            except (AttributeError, EOFError):
                # Handle case where manager connection is lost
                return False

            return True  # Continue the clock schedule

        # Schedule the update function to run periodically
        Clock.schedule_interval(update_progress, 0.05)  # Update 20 times/second


class SingleProgressBar(MDBoxLayout):
    indicator_color = ColorProperty([1, 0, 1, 1])
    value = NumericProperty(0.0)


class ProcessingProgressBar(MDBoxLayout):
    values = ListProperty([0])
    indicator_color = ColorProperty([1, 0, 1, 1])
    indicator_bg_color = ColorProperty([1, 0, 0, 1])

    def on_values(self, *args):
        values = self.values

        # Ensure the correct number of progress bars are present.
        while len(self.children) != len(values):
            if len(self.children) < len(values):
                self.add_widget(SingleProgressBar(
                    # md_bg_color=[0, 0, 1, 1],
                    # indicator_color=self.indicator_color,
                ))
            elif len(self.children) > len(values):
                self.remove_widget(self.children[-1])

        for i, progress_bar in enumerate(self.children):
            progress_bar.value = self.values[i]


class TimeIndicator(MDBoxLayout):
    time_elapsed = NumericProperty(0.0)
    processing_time = NumericProperty(0.0)


if __name__ == "__main__":
    App().run()