from IPython.display import display, clear_output
from ipywidgets import Button, VBox, HBox, Layout, Text, SelectMultiple, SelectionRangeSlider, Label

class MetricsSelection:
    def __init__(self, metrics_values):
        self.metrics_values = metrics_values
        self.query = ""
        self.selected_metrics = {
            "facilitiesServices": [],
            "priceRange": None,
            "cuisineType": [],
        }
        self.result = None

# ============================== QUERY ================================== #

    def query_section(self):
        self.query_input = Text(
            value="",
            placeholder="Enter query text here...",
            description="Query Text:",
            layout=Layout(width="auto"),
            style={"description_width": "auto"}
        )

        # Accept and Reset Buttons
        accept_button = Button(
            description="Accept Query",
            button_style="primary",
            icon="check",
            layout=Layout(width="150px")
        )

        def accept_text(change):
            self.query_input.disabled = True
            self.query = self.query_input.value
            # Disable button after click
            accept_button.disabled = True
            accept_button.button_style = ""

        accept_button.on_click(accept_text)

        reset_button = Button(
            description="Reset Query",
            button_style="warning",
            icon="refresh",
            layout=Layout(width="150px")
        )

        def reset_text(change):
            self.query_input.value = ""
            self.query_input.disabled = False
            self.query = ""
            # Enable button after click
            accept_button.disabled = False
            accept_button.button_style = "primary"

        reset_button.on_click(reset_text)

        action_buttons = HBox([accept_button, reset_button])

        # Section styling
        section = VBox(
            [Label("Query input: Enter your query (e.g. 'modern seasonal')"), self.query_input, action_buttons],
            layout=Layout(
                border="2px solid lightgray",
                padding="10px",
                margin="10px 0",
                border_radius="8px",
                background_color="#f9f9f9"
            )
        )

        return section

# =========================== FACILITIES ================================ #

    def facilities_selection(self):
        facilities = list(self.metrics_values["facilitiesServices"])
        self.selected_facilities = []

        # Text field to display selected facilities
        self.facilities_field = Text(
            value="",
            placeholder="Selected facilities will appear here...",
            description="Selected Facilities:",
            layout=Layout(width="auto"),
            style={"description_width": "auto"},
            disabled=True
        )

        # Toggle function for buttons
        def toggle_facility(button):
            facility = button.description
            if facility in self.selected_facilities:
                self.selected_facilities.remove(facility)
                button.button_style = ''  # Reset button style
            else:
                self.selected_facilities.append(facility)
                button.button_style = 'success'  # Mark as selected
            self.facilities_field.value = ", ".join(self.selected_facilities)

        # Create buttons for facilities
        self.facility_buttons = []
        for facility in facilities:
            button = Button(description=facility, button_style='', layout=Layout(width="auto", margin="5px"))
            button.on_click(toggle_facility)
            self.facility_buttons.append(button)

        buttons_inline = HBox(self.facility_buttons, layout=Layout(flex_flow="row wrap"))

        # Accept and Reset Buttons
        accept_button = Button(
            description="Accept Facilities",
            button_style="primary",
            icon="check",
            layout=Layout(width="150px", margin="5px")
        )

        def accept_facilities(change):
            self.selected_metrics["facilitiesServices"] = self.selected_facilities.copy()
            # Disable button after click
            accept_button.disabled = True
            accept_button.button_style = ""
            # Disable facilities after Accept
            for button in self.facility_buttons: 
                button.disabled = True 
                button.button_style = ''

        accept_button.on_click(accept_facilities)

        reset_button = Button(
            description="Reset Facilities",
            button_style="warning",
            icon="refresh",
            layout=Layout(width="150px", margin="5px")
        )

        def reset_facilities(change):
            for button in self.facility_buttons:
                button.disabled = False 
                button.button_style = ''  # Reset button styles
            self.selected_facilities.clear()
            self.facilities_field.value = ""
            self.selected_metrics["facilitiesServices"] = []
            # Enable button after click
            accept_button.disabled = False
            accept_button.button_style = "primary"

        reset_button.on_click(reset_facilities)

        action_buttons = HBox([accept_button, reset_button])

        # Section styling
        section = VBox(
            [Label("Facilities Selection: Click on service to add to the list"), self.facilities_field, buttons_inline, action_buttons],
            layout=Layout(
                border="2px solid lightgray",  # Add border
                padding="10px",  # Add padding inside the section
                margin="10px 0",  # Add margin between sections
                border_radius="8px",  # Add rounded corners
                background_color="#f9f9f9"  # Light gray background
            )
        )

        return section

# =========================== PRICE RANGE =============================== #

    def price_range_selection(self):
        price_range_options = self.metrics_values["priceRange"]

        # Text field to display selected price range
        self.price_range_field = Text(
            value="",
            placeholder="Selected ranges will appear here...",
            description="Selected Price Range:",
            layout=Layout(width="auto"),
            style={"description_width": "auto"},
            disabled=True
        )

        # Slider
        self.price_range_slider = SelectionRangeSlider(
            options=price_range_options,
            index=(0, len(price_range_options) - 1),
            description="Price Range:",
            layout=Layout(width="50%"),
            style={"description_width": "auto"}
        )

        # Update field on slider change
        def on_price_range_change(change):
            self.price_range_field.value = f"{change['new'][0]} - {change['new'][1]}"

        self.price_range_slider.observe(on_price_range_change, names='value')

        # Accept and Reset Buttons
        accept_button = Button(
            description="Accept Range",
            button_style="primary",
            icon="check",
            layout=Layout(width="150px")
        )

        def accept_range(change):
            self.selected_metrics["priceRange"] = self.price_range_slider.value
            # Disable button after click
            accept_button.disabled = True
            accept_button.button_style = ""
            # Disable slider
            self.price_range_slider.disabled = True

        accept_button.on_click(accept_range)

        reset_button = Button(
            description="Reset Range",
            button_style="warning",
            icon="refresh",
            layout=Layout(width="150px")
        )

        def reset_range(change):
            self.price_range_slider.value = (price_range_options[0], price_range_options[-1])
            self.price_range_field.value = "Price Range reset to default."
            self.selected_metrics["priceRange"] = None
            # Enable button after click
            accept_button.disabled = False
            accept_button.button_style = "primary"
            # Enable slider
            self.price_range_slider.disabled = False

        reset_button.on_click(reset_range)

        action_buttons = HBox([accept_button, reset_button])

        # Section styling
        section = VBox(
            [Label("Price Range Selection"), self.price_range_field, self.price_range_slider, action_buttons],
            layout=Layout(
                border="2px solid lightgray",
                padding="10px",
                margin="10px 0",
                border_radius="8px",
                background_color="#f9f9f9"
            )
        )

        return section

# ========================== CUISINE TYPE =============================== #

    def cuisine_selection(self):
        cuisine_options = list(self.metrics_values["cuisineType"])

        # Text field to display selected cuisine
        self.cuisine_field = Text(
            value="",
            placeholder="Selected cuisine types will appear here...",
            description="Selected Cuisine:",
            layout=Layout(width="auto"),
            style={"description_width": "auto"},
            disabled=True
        )

        # Multi-selection dropdown for cuisine selection
        self.cuisine_multi_select = SelectMultiple(
            options=cuisine_options,
            description='Cuisine:',
            style={"description_width": "auto"},
            layout=Layout(width='400px', height='100px')  # Adjust height for better visibility
        )

        # Accept and Reset Buttons
        accept_button = Button(
            description="Accept Cuisine",
            button_style="primary",
            icon="check",
            layout=Layout(width="150px")
        )

        def accept_cuisine(change):
            selected_cuisines = list(self.cuisine_multi_select.value)
            self.selected_metrics["cuisineType"] = selected_cuisines
            self.cuisine_field.value = f"{', '.join(selected_cuisines)}"
            # Disable button after click
            accept_button.disabled = True
            accept_button.button_style = ""
            # Disable list
            self.cuisine_multi_select.disabled = True

        accept_button.on_click(accept_cuisine)

        reset_button = Button(
            description="Reset Cuisine",
            button_style="warning",
            icon="refresh",
            layout=Layout(width="150px")
        )

        def reset_cuisine(change):
            self.cuisine_multi_select.value = ()  # Reset selection
            self.cuisine_field.value = ""
            self.selected_metrics["cuisineType"] = []
            # Enable button after click
            accept_button.disabled = False
            accept_button.button_style = "primary"
            # Enable list
            self.cuisine_multi_select.disabled = False

        reset_button.on_click(reset_cuisine)

        action_buttons = HBox([accept_button, reset_button])

        # Section styling
        section = VBox(
            [Label("Cuisine Type Selection: Multiple selection from the list"), self.cuisine_field, self.cuisine_multi_select, action_buttons],
            layout=Layout(
                border="2px solid lightgray",
                padding="10px",
                margin="10px 0",
                border_radius="8px",
                background_color="#f9f9f9"
            )
        )

        return section

# ============================ RENDERING ================================ #

    # Render Only Query Section
    def render_query(self):
        return VBox([
            self.query_section()
        ])
    
    # Render All Sections
    def render_all(self):
        return VBox([
            self.query_section(),
            self.facilities_selection(),
            self.price_range_selection(),
            self.cuisine_selection()
        ])


def create_simple_search_ui(search_function, *args, **kwargs):
    """
    Creates a search UI and dynamically executes the passed search function.

    Args:
        query_selector: The query selector widget.
        search_function: The function to execute for searching.
        *args: Positional arguments to pass to the search function.
        **kwargs: Keyword arguments to pass to the search function.

    Returns:
        None
    """
    # Define query selector
    query_selector = MetricsSelection(None)
    # Display the query selector
    display(query_selector.render_query())

    # Create the Search button
    search_button = Button(
        description="Search",
        button_style="primary",  # Blue button
        icon="search",  # Add a search icon
        layout={"width": "150px"}
    )

    # Define Widget Event that will display the result of ranking matching restaurants
    def search_ranked(change):
        # Clear the entire cell's output
        clear_output(wait=True)
        
        # Re-render the query interface and search button
        display(query_selector.render_query())
        display(search_button)

        # Fetch the query
        query = query_selector.query
        print(f"Query text: {query}")
        if query:
            # Dynamically execute the passed search function with args/kwargs
            result = search_function(query, *args, **kwargs)
            display(result)
        else:
            display("ENTER QUERY TEXT, BITTE!")

    # Attach the search function to the button
    search_button.on_click(search_ranked)

    # Display the Search button
    display(search_button)

def create_metrics_search_ui(metrics_selector, search_function, *args, **kwargs):
    """
    Creates a search UI and dynamically executes the passed search function.

    Args:
        metrics_values: The query selector widget.
        search_function: The function to execute for searching.
        *args: Positional arguments to pass to the search function.
        **kwargs: Keyword arguments to pass to the search function.

    Returns:
        None
    """
    # Display the query selector
    display(metrics_selector.render_all())

    # Create the Search button
    search_button = Button(
        description="Search",
        button_style="primary",  # Blue button
        icon="search",  # Add a search icon
        layout={"width": "150px"}
    )

    result = None

    # Define Widget Event that will display the result of ranking matching restaurants
    def search_ranked(change):
        # Clear the entire cell's output
        clear_output(wait=True)
        
        # Re-render the query interface and search button
        display(metrics_selector.render_all())
        display(search_button)

        # Fetch the query and metrics
        query = metrics_selector.query
        selected_metrics = metrics_selector.selected_metrics
        print(f"Query text: {query}")
        print(f"Selected metrics: {selected_metrics}")
        if query:
            # Dynamically execute the passed search function with args/kwargs
            result = search_function(query, selected_metrics, *args, **kwargs)
            display(result)
            metrics_selector.result = result
        else:
            display("ENTER QUERY TEXT, BITTE!")

    # Attach the search function to the button
    search_button.on_click(search_ranked)

    # Display the Search button
    display(search_button)

