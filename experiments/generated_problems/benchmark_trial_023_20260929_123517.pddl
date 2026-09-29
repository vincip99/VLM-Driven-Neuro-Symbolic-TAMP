(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    bright_yellow_block - obj
    cooking_pot - location
  )
  (:init
    (on-table bright_yellow_block)
  )
  (:goal (and
    (on bright_yellow_block cooking_pot)
  ))
)
