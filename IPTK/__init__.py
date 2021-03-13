from window_init import Window


if __name__ == '__main__':
    ipkt = Window()
    window = ipkt.root
    ipkt.create_menu(window)
    window.mainloop()
