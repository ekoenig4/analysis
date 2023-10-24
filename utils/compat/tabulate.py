class tabulate:
    def __init__(self, table, headers=[], **kwargs):
        self.__dict__.update(kwargs)
        self.table = table
        self.headers = headers

    def format_value(self, value):
        if isinstance(value, float) and hasattr(self, 'floatfmt'):
            return f'{value:{self.floatfmt}}'
        return value

    def __str__(self):
        spaces  = [
            max( len(str( self.format_value(value) )) for value in column )
            for column in zip(self.headers, *self.table)
        ]

        string = ' '
        for header, space in zip(self.headers, spaces):
            string += header.center(space) + ' '
        string += '\n'

        string += ' '
        for space in spaces:
            string += '-' * space + ' '
        string += '\n'

        for row in self.table:
            string += ' '
            for value, space in zip(row, spaces):
                if isinstance(value, (int, float)) and hasattr(self, 'numalign'):
                    if self.numalign == 'decimal':
                        value = f'{self.format_value(value):>{space}}'
                    elif self.numalign == 'center':
                        value = f'{self.format_value(value):^{space}}'
                    elif self.numalign == 'left':
                        value = f'{self.format_value(value):<{space}}'
                    elif self.numalign == 'right':
                        value = f'{self.format_value(value):>{space}}'
                    else:
                        raise ValueError('invalid alignment')
                else:
                    value = f'{value:<{space}}'

                string += value + ' '
            string += '\n'
        return string
            