import logging
import logging.handlers


class LineCountRotatingHandler(logging.Handler):
    def __init__(self, base_filename, max_lines, max_bytes, backup_count):
        super().__init__()
        self.base_filename = base_filename
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.rotating_handler = logging.handlers.RotatingFileHandler(
            base_filename, maxBytes=max_bytes, backupCount=backup_count
        )
        self.line_count = 0

    def emit(self, record):
        # Write log entry and rotate by line count if necessary
        log_entry = self.format(record)
        with open(self.base_filename, 'a') as log_file:
            log_file.write(log_entry + '\n')
            self.line_count += 1

        # If we exceed the max line count, we rotate (delete the oldest lines)
        if self.line_count >= self.max_lines:
            self._truncate_log_file()
        
        # Also check if we need to rotate based on file size (delegated to RotatingFileHandler)
        # self.rotating_handler.emit(record)

    def _truncate_log_file(self):
        """Truncate the log file to maintain a maximum number of lines."""
        with open(self.base_filename, 'r') as log_file:
            lines = log_file.readlines()

        # Keep only the last `max_lines` lines
        with open(self.base_filename, 'w') as log_file:
            log_file.writelines(lines[-self.max_lines:])

        # Reset the line count
        self.line_count = self.max_lines

    def close(self):
        self.rotating_handler.close()
        super().close()

def setup_logger(logger_name, log_path):
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
  
    # Create custom handler
    max_lines = 500       # Limit to 500 lines
    max_file_size = 5 * 1024 * 1024  # Limit to 5 MB
    backup_count = 5       # Keep 5 backup log files

    handler = LineCountRotatingHandler(log_path, max_lines, max_file_size, backup_count)
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger
    
