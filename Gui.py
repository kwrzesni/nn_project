from tkinter import *
from tkinter import messagebox
from NeuralNetwork import NeuralNetwork
from matplotlib import pyplot as plt
import numpy as np


class TrainingWindow:
    def __init__(self, epochs, training_history):
        self.frame = Tk()
        self.frame.resizable(0, 0)
        self.frame.title('Przebieg uczenia sieci')
        screen_width = self.frame.winfo_screenwidth()
        screen_height = self.frame.winfo_screenheight()
        self.frame.geometry('325x235+%d+%d' % ((screen_width - 325) / 2, (screen_height - 235) / 2))
        self.frame.grid_propagate(False)
        self.frame.eval('tk::PlaceWindow . center')

        self.epochs = epochs
        self.training_history = training_history

        self.__create_widgets()

        self.frame.mainloop()

    def __create_widgets(self):
        self.training_label = Label(self.frame, text='Metoda uczenie:     Spadek wzdłuż gradientu z momentem')
        self.training_label.place(x=5, y=5)

        self.cost_label = Label(self.frame, text='Funkcja kosztu:       Błąd średniokwadratowy')
        self.cost_label.place(x=5, y=25)

        self.epochs_label = Label(self.frame, text='Liczba epok:            {:d}'.format(self.epochs))
        self.epochs_label.place(x=5, y=45)

        self.plot_performance_button = Button(self.frame, text='Wykres trafności sieci', bg='#E1E1E1',
                                              command=lambda: self.__plot_performance())
        self.plot_performance_button.place(x=5, y=120, width=315, height=50)

        self.plot_loss_button = Button(self.frame, text='Wykres kosztu sieci', bg='#E1E1E1',
                                       command=lambda: self.__plot_loss())
        self.plot_loss_button.place(x=5, y=180, width=315, height=50)

    def __plot_performance(self):
        history_dict = self.training_history.history

        plt.figure()
        plt.plot(history_dict['accuracy'])
        plt.xlabel('{:d} Epok'.format(self.epochs))
        plt.ylabel('trafności')
        plt.title('Wykres trafności sieci')
        plt.grid(True)
        plt.show()

    def __plot_loss(self):
        history_dict = self.training_history.history

        plt.figure()
        plt.plot(history_dict['loss'])
        plt.xlabel('{:d} Epok'.format(self.epochs))
        plt.ylabel('Błąd średniokwadratowy')
        plt.title('Wykres błędu sieci')
        plt.grid(True)
        plt.show()


class Gui:
    def __init__(self):
        self.frame = Tk()
        self.frame.resizable(0, 0)
        self.frame.title('gui')
        screen_width = self.frame.winfo_screenwidth()
        screen_height = self.frame.winfo_screenheight()
        self.frame.geometry('770x395+%d+%d' % ((screen_width - 770) / 2, (screen_height - 395) / 2))
        self.frame.grid_propagate(False)

        self.__create_widgets()

        self.network = None

        self.training_window = None

        self.frame.mainloop()

    def __create_widgets(self):
        self.__create_create_network_widgets()
        self.__create_learn_network_widgets()
        self.__create_additional_point_widgets()
        self.__create_test_network_widgets()

        self.response_surface_button = Button(self.frame, text='Powierzchnia odpowiedzi', bg='#E1E1E1',
                                              command=lambda: self.__plot_response_surface())
        self.response_surface_button.place(x=515, y=202, width=250, height=50)

        self.switch_function_button = Button(self.frame, text='Funkcja przełączająca', bg='#E1E1E1',
                                             command=lambda: self.__plot_switching_function())
        self.switch_function_button.place(x=515, y=260, width=250, height=50)

        self.__create_network_widgets()

    def __create_create_network_widgets(self):
        self.create_network_frame = LabelFrame(self.frame, text='Tworzenie sieci', width=250, height=305)
        self.create_network_frame.place(x=5, y=5)

        self.__create_learning_data_widgets()
        self.__create_activation_widgets()

        self.create_network_button = Button(self.create_network_frame, text='Utwórz sieć', bg='#E1E1E1',
                                            command=lambda: self.__create_network())
        self.create_network_button.place(x=5, y=200, width=235, height=80)

    def __create_learning_data_widgets(self):
        self.learn_data_frame = LabelFrame(self.create_network_frame, text='Dane uczące', width=140, height=190)
        self.learn_data_frame.place(x=5, y=5)

        self.learning_data_input_label = Label(self.learn_data_frame, text='Wejścia')
        self.learning_data_input_label.place(x=10, y=5, width=65)

        self.learning_data_input_entries = []
        self.learning_data_input_var = [[StringVar() for _ in range(2)] for _ in range(4)]
        default_input_entry_values = [['0', '0'], ['0', '1'], ['1', '0'], ['1', '1']]
        for row in range(4):
            entries = []
            for col in range(2):
                entry = Entry(self.learn_data_frame, justify=CENTER, textvariable=self.learning_data_input_var[row][col])
                entry.insert(0, default_input_entry_values[row][col])
                entry.place(x=10+col*35, y=30+row*35, width=30, height=30)
                entries.append(entry)
            self.learning_data_input_entries.append(entries)

        self.learning_data_output_label = Label(self.learn_data_frame, text='Wyjście')
        self.learning_data_output_label.place(x=85, y=5, width=47)

        self.learning_data_output_entries = []
        self.learning_data_output_var = [StringVar() for _ in range(4)]
        default_output_entry_values = ['0', '0', '0', '1']
        for row in range(4):
            entry = Entry(self.learn_data_frame, justify=CENTER, textvariable=self.learning_data_output_var[row])
            entry.insert(0, default_output_entry_values[row])
            entry.place(x=95, y=30 + row * 35, width=30, height=30)
            self.learning_data_output_entries.append(entry)

    def __create_activation_widgets(self):
        self.activation_frame = LabelFrame(self.create_network_frame, text='F. aktywacji', width=90, height=190)
        self.activation_frame.place(x=150, y=5)

        self.activation_function_ind = 0
        self.activations = ['linear', 'step', 'sigmoid']
        self.activation_var = IntVar()
        self.radio_button_linear = Radiobutton(self.activation_frame, text='liniowa',
                                               variable=self.activation_var, value=0)
        self.radio_button_linear.place(x=5, y= 5)

        self.radio_button_threshold = Radiobutton(self.activation_frame, text='progowa',
                                                  variable=self.activation_var, value=1)
        self.radio_button_threshold.place(x=5, y=25)

        self.radio_button_sigmoid = Radiobutton(self.activation_frame, text='sigmoida',
                                                variable=self.activation_var, value=2)
        self.radio_button_sigmoid.place(x=5, y=45)

    def __create_learn_network_widgets(self):
        self.learn_network_frame = LabelFrame(self.frame, text='Uczenie sieci', width=250, height=185)
        self.learn_network_frame.place(x=260, y=5)

        self.number_of_epochs_label = Label(self.learn_network_frame, text='Liczba epok:')
        self.number_of_epochs_label.place(x=10, y=5, height=30)

        self.number_of_epochs_var = StringVar()
        self.number_of_epochs_entry = Entry(self.learn_network_frame, justify=CENTER,
                                            textvariable=self.number_of_epochs_var)
        self.number_of_epochs_entry.insert(0, '250')
        self.number_of_epochs_entry.place(x=150, y=5, width=85, height=30)

        self.learning_rate_label = Label(self.learn_network_frame, text='Współczynnik uczenia:')
        self.learning_rate_label.place(x=10, y=40, height=30)

        self.learning_rate_var = StringVar()
        self.learning_rate_entry = Entry(self.learn_network_frame, justify=CENTER,
                                         textvariable=self.learning_rate_var)
        self.learning_rate_entry.insert(0, '0.1')
        self.learning_rate_entry.place(x=150, y=40, width=85, height=30)

        self.momentum_label = Label(self.learn_network_frame, text='Bezwładność:')
        self.momentum_label.place(x=10, y=75, height=30)

        self.momentum_var = StringVar()
        self.momentum_entry = Entry(self.learn_network_frame, justify=CENTER, textvariable=self.momentum_var)
        self.momentum_entry.insert(0, '0.1')
        self.momentum_entry.place(x=150, y=75, width=85, height=30)

        self.learn_network_button = Button(self.learn_network_frame, text='Naucz sieć', bg='#E1E1E1',
                                           command=lambda: self.__train_network())
        self.learn_network_button.place(x=10, y=110, width=225, height=50)

    def __create_additional_point_widgets(self):
        self.additional_point_frame = LabelFrame(self.frame, text='Dodatkowy punkt', width=250, height=115)
        self.additional_point_frame.place(x=260, y=195)

        self.additional_point_var = IntVar()
        self.additional_point_checkbox = Checkbutton(self.additional_point_frame, text='Dodatkowy punkt',
                                                     variable=self.additional_point_var)
        self.additional_point_checkbox.place(x=5, y=5)

        self.additional_point_input_label = Label(self.additional_point_frame, text='Wejścia')
        self.additional_point_input_label.place(x=10, y=35, width=65)

        self.additional_point_input_entries = []
        self.additional_point_input_var = [StringVar(), StringVar()]
        for col in range(2):
            entry = Entry(self.additional_point_frame, justify=CENTER,
                          textvariable=self.additional_point_input_var[col])
            entry.insert(0, '0')
            entry.place(x=10 + col * 35, y=60, width=30, height=30)
            self.additional_point_input_entries.append(entry)

        self.additional_point_output_label = Label(self.additional_point_frame, text='Wyjście')
        self.additional_point_output_label.place(x=199, y=35, width=47)

        self.additional_point_output_var = StringVar()
        self.additional_point_output_entry = Entry(self.additional_point_frame, justify=CENTER,
                                                   textvariable=self.additional_point_output_var)
        self.additional_point_output_entry.insert(0, '0')
        self.additional_point_output_entry.place(x=205, y=60, width=30, height=30)

    def __create_test_network_widgets(self):
        self.test_network_frame = LabelFrame(self.frame, text='Testowanie sieci', width=250, height=185)
        self.test_network_frame.place(x=515, y=5)

        self.test_network_input_label = Label(self.test_network_frame, text='Wejścia')
        self.test_network_input_label.place(x=5, y=10, width=65)

        self.test_network_input_entries = []
        self.test_network_input_var = [StringVar(), StringVar()]
        for col in range(2):
            entry = Entry(self.test_network_frame, justify=CENTER,
                          textvariable=self.test_network_input_var[col])
            entry.insert(0, '0')
            entry.place(x=5 + col * 35, y=35, width=30, height=30)
            self.test_network_input_entries.append(entry)

        self.test_network_output_label = Label(self.test_network_frame, text='Wyjście')
        self.test_network_output_label.place(x=165, y=10, width=75)

        self.test_network_output_var = StringVar()
        self.test_network_output_entry = Entry(self.test_network_frame, justify=CENTER,
                                               textvariable=self.test_network_output_var,
                                               state='readonly', readonlybackground='#FFFFFF')
        self.test_network_output_entry.place(x=165, y=35, width=75, height=30)

        self.test_network_button = Button(self.test_network_frame, text='Testuj wyjście', bg='#E1E1E1',
                                          command=lambda: self.__test_network())
        self.test_network_button.place(x=5, y=110, width=235, height=50)

    def __create_network_widgets(self):
        self.network_frame = LabelFrame(self.frame, text='Sieć', width=760, height=80)
        self.network_frame.place(x=5, y=310)

        self.network_weight_label = Label(self.network_frame, text='Wagi neuronów:')
        self.network_weight_label.place(x=245, y=5, width=170)

        self.network_weight_entries = []
        self.network_weight_var = [StringVar(), StringVar()]
        for col in range(2):
            entry = Entry(self.network_frame, justify=CENTER,
                          textvariable=self.network_weight_var[col], state='readonly', readonlybackground='#FFFFFF')
            entry.place(x=245 + col * 90, y=25, width=80, height=30)
            self.network_weight_entries.append(entry)

        self.network_weight_label = Label(self.network_frame, text='Bias:')
        self.network_weight_label.place(x=435, y=5, width=80)

        self.network_bias_var = StringVar()
        self.network_bias_entry = Entry(self.network_frame, justify=CENTER,
                                        textvariable=self.network_bias_var,
                                        state='readonly', readonlybackground='#FFFFFF')
        self.network_bias_entry.place(x=435, y=25, width=80, height=30)

    def __create_network(self):
        self.activation_function_ind = self.activation_var.get()
        self.network = NeuralNetwork(self.activation_function_ind)
        self.__update_weight_and_bias()

    def __train_network(self):
        if self.network is None:
            messagebox.showerror('Sieć nie istnieje', 'Przed rozpoczęciem trenowania sieci należy ją utworzyć')
            return

        X = []
        y = []
        for row in range(4):
            X.append([float(self.learning_data_input_var[row][0].get()),
                      float(self.learning_data_input_var[row][1].get())])
            y.append(float(self.learning_data_output_var[row].get()))

        if self.additional_point_var.get():
            X.append([float(self.additional_point_input_var[0].get()),
                      float(self.additional_point_input_var[1].get())])
            y.append(float(self.additional_point_output_var.get()))

        epochs = int(self.number_of_epochs_var.get())
        learning_rate = float(self.learning_rate_var.get())
        momentum = float(self.momentum_var.get())

        self.network.train(X, y, epochs, learning_rate, momentum)
        self.__update_weight_and_bias()

        self.training_window = TrainingWindow(epochs, self.network.training_history)

    def __update_weight_and_bias(self):
        weight = self.network.get_weight()
        self.network_weight_var[0].set('{:.4f}'.format(weight[0]))
        self.network_weight_var[1].set('{:.4f}'.format(weight[1]))

        bias = self.network.get_bias()
        self.network_bias_var.set('{:.4f}'.format(bias))

    def __test_network(self):
        if self.network is None:
            messagebox.showerror('Sieć nie istnieje', 'Przed rozpoczęciem testowania sieci należy ją utworzyć')
            return

        X = [float(self.test_network_input_var[0].get()), float(self.test_network_input_var[1].get())]
        y = self.network.predict(X)

        self.test_network_output_var.set('{:.4f}'.format(y))

    def __plot_switching_function(self):
        if self.network is None:
            messagebox.showerror('Sieć nie istnieje', 'Przed narysowaniem funkcji przełączającej należy utworzyć sieć')
            return

        w1, w2 = self.network.get_weight()
        bias = self.network.get_bias()
        a = -w1/w2
        b = -bias/w2
        x = np.linspace(0, 1, 100)
        y = a*x+b

        plt.figure()
        plt.plot(x, y)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('x2 = {:.5f} * x1 + ({:.5f})'.format(a, b))
        plt.grid(True)
        plt.axis([0, 1, 0, 1])
        plt.show()

    def __plot_response_surface(self):
        if self.network is None:
            messagebox.showerror('Sieć nie istnieje', 'Przed narysowaniem powierzchni odpowiedzi należy utworzyć sieć')
            return

        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)

        w1, w2 = self.network.get_weight()
        bias = self.network.get_bias()

        Z = self.network.activations_list[self.activation_function_ind](w1 * X + w2 * Y + bias)

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('Wejście 1')
        ax.set_ylabel('Wejście 2')
        ax.set_zlabel('Odpowiedź')
        ax.set_title('Powierzchnia odpowiedzi')
        ax.grid(True)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.invert_xaxis()
        plt.show()