// DataTables
import 'datatables.net-bs4';
import 'datatables.net-select-bs4'
import 'datatables.net-buttons-bs4'

(function ($) {
    "use strict"; // Start of use strict

    $(document).ready(function () {
        let table = $('#rationalesDataTable').DataTable({
            deferRender: true,
            stateSave: true,
            select: {
                style: 'single',
            },
            ajax: $AJAX_RATIONALES_URL,
            columns: [
                {title: '#', data: 'id'},
                {title: 'User', data: 'user'},
                {title: 'Text', data: 'text'},
                {title: 'Tokens', data: 'tokens'},
            ],
            buttons: [
                {
                    extend: 'selectedSingle',
                    text: '<i class="fas fa-eye"></i> Show',
                    action: function (e, dt, button, config) {
                        window.location.href = dt.row({selected: true}).data()['show_url'];
                    }
                },
                {
                    extend: 'selectedSingle',
                    text: '<i class="fas fa-edit"></i> Edit',
                    action: function (e, dt, button, config) {
                        window.location.href = dt.row({selected: true}).data()['edit_url'];
                    }
                },
                {
                    extend: 'selectedSingle',
                    text: '<i class="fas fa-trash"></i> Delete',
                    action: function (e, dt, button, config) {
                        $("#delete-form").attr('action', dt.row({selected: true}).data()['delete_url']).submit();
                    }
                }
            ],
            initComplete: function (settings, json) {
                table.buttons().container().appendTo($('#rationalesDataTable_wrapper .col-md-6').eq(0));
            }
        });
    });

})(jQuery); // End of use strict