.PHONY: all
all: doc
	
.PHONY: doc
doc:
	sphinx-apidoc -fF -o ./docs ./pyutils

.PHONY: test
test:
	pytest

.PHONY: clean
clean:
	$(RM) -r ./docs
